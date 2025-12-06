# -*- coding: utf-8 -*-
import onnxruntime as ort
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, RobertaTokenizer

# ================= 配置区域 =================
ONNX_MODEL_PATH = "mnre_model.onnx"
# ⚠️ 这里必须改成你训练时的 text_encoder 路径或名称
TEXT_ENCODER = "roberta-base"  # 或者是 "roberta-base"

# ⚠️ 这里的 ID 映射必须与训练时的 dataset.label_mapping 完全一致！
# 请查看训练日志中的 "Detected labels"
ID2LABEL = {
    0: "PAD",
    1: "O",
    2: "B-MISC", 3: "I-MISC",
    4: "B-PER",  5: "I-PER",
    6: "B-ORG",  7: "I-ORG",
    8: "B-LOC",  9: "I-LOC",
    10: "X",
    11: "[CLS]", 12: "[SEP]"
}
MAX_LEN = 128
# ===========================================

class ONNXPredictor:
    def __init__(self, model_path, text_encoder_name):
        print(f"正在加载 ONNX 模型: {model_path} ...")
        # 加载 ONNX 推理引擎
        # providers=['CUDAExecutionProvider'] 如果有 GPU，否则用 ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 加载分词器
        print(f"正在加载分词器: {text_encoder_name} ...")
        if "roberta" in text_encoder_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(text_encoder_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(text_encoder_name)

        # 定义图像预处理 (必须与训练时 dataloader.py 一致)
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, text, image_path=None):
        """将文本和图片转换为 ONNX 需要的 NumPy 数组"""
        
        # --- 1. 文本处理 ---
        # 简单处理：不像训练那样做复杂的 label 对齐，直接 encode
        # 实际部署时建议按字切分(中文)或 Subword(英文)
        tokens = self.tokenizer.tokenize(text)
        # 截断
        if len(tokens) > MAX_LEN - 2:
            tokens = tokens[:MAX_LEN - 2]
            
        encode_dict = self.tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='np'  # 直接返回 numpy
        )
        
        input_ids = encode_dict['input_ids'].astype(np.int64)
        attention_mask = encode_dict['attention_mask'].astype(np.int64)

        # --- 2. 图像处理 ---
        if image_path:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.img_transform(image)
                # 增加 batch 维度 [1, 3, 224, 224] 并转为 numpy
                image_numpy = image_tensor.unsqueeze(0).numpy()
            except Exception as e:
                print(f"图片读取失败: {e}，使用全零占位符")
                image_numpy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        else:
            # 如果没有图片，必须给一个符合维度的 Dummy 输入
            image_numpy = np.zeros((1, 3, 224, 224), dtype=np.float32)

        return input_ids, attention_mask, image_numpy, tokens

    def predict(self, text, image_path=None):
        # 1. 预处理
        input_ids, attention_mask, image_numpy, raw_tokens = self.preprocess(text, image_path)
        
        # 2. 构建输入字典
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image_numpy # 确保这里和你导出时的 input_names 一致
        }
        
        # 3. 执行推理
        outputs = self.session.run(None, onnx_inputs)
        model_output = outputs[0]
        
        # 处理输出维度
        if len(model_output.shape) == 3:
            pred_ids = np.argmax(model_output, axis=-1)[0]
        else:
            pred_ids = model_output[0]
            
        # 4. 解码与结果重组
        results = []
        valid_len = np.sum(attention_mask[0])
        
        print(f"\n📝 文本: {text}")
        print("-" * 50)
        print(f"{'Token':<15} | {'Label':<10} | {'Cleaned'}")
        print("-" * 50)
        
        current_entity = None
        
        # 跳过 [CLS] 和 [SEP]
        for i in range(1, valid_len - 1):
            if i >= len(pred_ids): break
            
            pid = pred_ids[i]
            label = ID2LABEL.get(pid, "O")
            
            # 获取 token 并清洗 RoBERTa 的特殊字符 Ġ
            raw_token = raw_tokens[i-1] if (i-1) < len(raw_tokens) else ""
            clean_token = raw_token.replace("Ġ", "") # 去掉 RoBERTa 的前导空格标记
            
            print(f"{raw_token:<15} | {label:<10} | {clean_token}")
            
            # === 核心逻辑修改开始 ===
            
            # 情况 1: 新实体开始 (B-xxx)
            if label.startswith("B-"):
                if current_entity: results.append(current_entity)
                current_entity = {
                    "type": label.split("-")[1], 
                    "word": clean_token
                }
            
            # 情况 2: 实体继续 (I-xxx)
            elif label.startswith("I-") and current_entity:
                if label.split("-")[1] == current_entity["type"]:
                    # 智能拼接：如果 raw_token 有 Ġ，说明它是新单词的开始，加空格；否则直接拼
                    if raw_token.startswith("Ġ"):
                         current_entity["word"] += " " + clean_token
                    else:
                         current_entity["word"] += clean_token
                else:
                    results.append(current_entity)
                    current_entity = None
            
            # 情况 3: 子词延续 (X) - 这就是之前丢失 "on" 和 "plex" 的原因
            elif label == "X" and current_entity:
                # X 通常是子词（如 ##ing 或 plex），直接拼接，不加空格
                current_entity["word"] += clean_token
                
            # 情况 4: 非实体 (O)
            else:
                if current_entity: results.append(current_entity)
                current_entity = None
                
            # === 核心逻辑修改结束 ===

        if current_entity: results.append(current_entity)
        
        print("-" * 50)
        print("🔍 最终抽取结果:")
        for res in results:
            print(f"  - [{res['type']}] {res['word']}")

if __name__ == "__main__":
    # 初始化预测器
    predictor = ONNXPredictor(ONNX_MODEL_PATH, TEXT_ENCODER)
    
    # 测试案例 1 (带图)
    text1 = "Apple is reportedly in talks to acquire AI search startup Perplexity, valued at $1.4 billion, as part of its generative AI push."
    # text1 = "我在北京天安门看升旗" # 如果是中文模型
    img1 = "data/ner_img/2138375.jpg" # 换成你存在的图片路径
    predictor.predict(text1, img1)
    
    # 测试案例 2 (无图)
    text2 = "Elon Musk is the CEO of Tesla."
    predictor.predict(text2, None)