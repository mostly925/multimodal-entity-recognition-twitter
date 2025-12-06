# -*- coding: utf-8 -*-
import torch
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizer

from model import build_model, _resolve_path 
from config import get_config
from test import load_config

script_dir = os.path.dirname(os.path.abspath(__file__))

def visualize_attention(save_name, text_input, image_input):
    """
    可视化文本对图像的注意力权重热力图
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    try:
        config = load_config(save_name)
    except:
        config = get_config()
        config.num_labels = 20 # 占位
    
    # 强制开启 output_attn 开关（虽然我们也会通过参数传）
    config.output_attn = True
    config.device = str(device)
    
    model = build_model(config).to(device)
    model_path = os.path.join(script_dir, "save_models", save_name, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功")
    else:
        print("使用未训练模型进行演示...")
        
    model.eval()
    
    # 2. 准备数据处理器
    t_path = _resolve_path(script_dir, config.text_encoder)
    tokenizer = RobertaTokenizer.from_pretrained(t_path)
    
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 处理输入
    # 文本
    tokens = tokenizer.tokenize(text_input)
    # 添加 CLS 和 SEP
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.encode(text_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # 图像
    if os.path.exists(image_input):
        raw_image = Image.open(image_input).convert('RGB')
        image_tensor = img_transform(raw_image).unsqueeze(0).to(device)
    else:
        print("图片路径不存在，使用随机噪声代替")
        image_tensor = torch.randn(1, 3, 224, 224).to(device)

    # 4. 前向传播获取 Attention
    print("正在推理...")
    with torch.no_grad():
        # 调用 model.forward，开启 return_weights=True
        # 注意：forward 返回的是 (total_loss, attn_weights) 或者 logits
        # 我们的修改让它在 return_weights=True 时返回 (loss, attn_weights)
        # 这里为了简化，我们假设 labels=None 时它走推理分支，但我们需要它的内部权重
        # 所以我们直接调用内部逻辑或者确保 forward 支持
        
        # 直接调用 forward，传入 return_weights=True
        # 根据我们修改后的 model.py，如果 labels=None 但 return_weights=True，它会返回 (total_loss, attn_weights)
        # 其中的 total_loss 可能是 0
        _, attn_weights = model(
            input_ids, 
            attention_mask, 
            image_tensor=image_tensor,
            return_weights=True
        )
    
    # attn_weights 形状通常是 [Batch, Seq_Len, Img_Tokens] (PyTorch Multihead 默认平均了 heads)
    # 或者 [Batch, Heads, Seq_Len, Img_Tokens] 取决于实现细节
    # 在我们的 CrossAttentionBlock 中，attn_weights 是 self.attn 返回的第二个值
    # PyTorch 默认: [Batch, Seq_Len, K_img] (average over heads)
    
    if attn_weights is None:
        print("未获取到注意力权重，请检查 model.py 是否正确修改")
        return

    # [Seq_Len, K_img]
    attn_map = attn_weights[0].cpu().numpy()
    
    # 5. 绘图
    plt.figure(figsize=(10, 8))
    # y轴标签：文本 Token
    # x轴标签：视觉 Token (Resampler 产生的 8 个摘要 Token)
    sns.heatmap(attn_map, xticklabels=[f"Vis_{i}" for i in range(attn_map.shape[1])], yticklabels=tokens)
    plt.title("Cross-Modal Attention: Text queries Image")
    plt.xlabel("Visual Resampler Tokens")
    plt.ylabel("Text Tokens")
    
    save_path = "attention_heatmap.png"
    plt.savefig(save_path)
    print(f"🔥 热力图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="模型目录名")
    parser.add_argument("--text", type=str, default="Apple is looking at buying U.K. startup for $1 billion", help="测试文本")
    parser.add_argument("--img", type=str, default="data/ner_img/10.jpg", help="测试图片路径")
    args = parser.parse_args()
    
    visualize_attention(args.save_name, args.text, args.img)