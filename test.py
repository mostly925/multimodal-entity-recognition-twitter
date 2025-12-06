import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入自定义的评估函数，用于计算精确率、召回率和 F1 值
from metrics import evaluate_each_class, evaluate
# 导入数据加载器和处理器，用于处理多模态 NER 数据
from dataloader import NERDataset, DataProcessor
# 导入模型构建函数
from model import build_model

script_dir = os.path.dirname(os.path.abspath(__file__))


# ========= 工具：Span → BIO ids（专用版） =========
def spans_to_bio_ids(seq_len, spans, label_mapping):
    """
    将 span 解码结果转换为 BIO id 序列
    spans: [(s, e, type, score), ...]  e 为右开区间
    """
    # 定义实体类型 ID 到字符串的映射
    TYPE_ID2STR = {0: "LOC", 1: "ORG", 2: "OTHER", 3: "PER"}
    # 初始化一个长度为 seq_len 的列表,所有位置先默认填充为 "O" (Outside) 标签对应的 ID
    # 这里的 seq_len 是当前样本的有效长度
    bio = [label_mapping["O"]] * seq_len
    
    # 定义一个辅助函数，把实体类型字符串转为 B/I 标签字符串
    def type_to_tags(tstr):
        # 特殊处理，"OTHER" 类型对应 "MISC" 标签，这是数据集的约定
        if tstr == "OTHER":
            return "B-MISC", "I-MISC"
        return f"B-{tstr}", f"I-{tstr}"
    # 遍历模型预测出的每一个 span
    # s=start, e=end, t=type, *_ 忽略掉后面的 score 等元素
    # spans 列表通常是按得分排序的，或者模型输出的原始顺序
    for s, e, t, *_ in spans:
        if isinstance(t, int):# 如果 t 是整数
            # 将整数类型 ID 转换为字符串类型
            tstr = TYPE_ID2STR.get(t, None)
        else:# 如果 t 本身就是字符串，直接用
            tstr = t
        if tstr is None:
            continue
        # 确保 span 的范围在序列长度内，且 start < end
        if not (0 <= s < e <= seq_len):
            continue
        # # 调用 type_to_tags("PER")，获取 B/I 标签的字符串
        # btag = "B-PER", itag = "I-PER"
        btag, itag = type_to_tags(tstr)
        # 从 label_mapping 中查出 B/I 标签对应的 ID
        # 如果找不到对应的标签，默认使用 "O"
        b_id = label_mapping.get(btag, label_mapping["O"])
        i_id = label_mapping.get(itag, label_mapping["O"])
        # 最关键的一步：在 bio 列表中填入 B/I 标签的 ID
        # 将 span 的起始位置标记为 B-Tag
        bio[s] = b_id
        # 将 span 的后续位置标记为 I-Tag
        for i in range(s + 1, e):
            bio[i] = i_id
    return bio


def evaluate_model(model, val_loader, device, tags):
    # 切换模型到评估模式，关闭 dropout 等
    model.eval()
    all_preds, all_labels, all_words = [], [], []
    # 创建一个 ID 到标签名的反向映射
    idx2tag = {v: k for k, v in tags.items()}

    with torch.no_grad():
        # 循环处理每个批次
        for batch in val_loader:
            # 把数据从 CPU 搬到 GPU
            # input_ids: 文本输入的 token ID
            input_ids = batch[0].to(device, non_blocking=True)
            # attention_mask: 标记有效 token 的掩码
            attention_mask = batch[1].to(device, non_blocking=True)
            # labels: 真实的 BIO 标签 ID
            labels = batch[2].to(device, non_blocking=True)
            # image_tensor: 图像特征
            image_tensor = batch[3].to(device, non_blocking=True)

            # —— Span 模式：调用模型进行预测，得到的是 Span 列表，再转 BIO ids
            # predict_spans 返回每个样本预测出的 span 列表
            span_lists = model.predict_spans(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                topk_s=8, topk_e=8
            )
            preds_bio = []
            for b in range(len(span_lists)):
                # 获取当前样本的有效长度
                valid_len = int(attention_mask[b].sum().item())
                # 取出当前句子的 span 预测
                spans_be_t = [(s, e, t, sc) for (s, e, t, sc) in span_lists[b]]
                # 将 Span 格式翻译成 BIO ID 序列
                bio_ids = spans_to_bio_ids(valid_len, spans_be_t, tags)
                # pad 对齐到序列长度
                # 因为 batch 中的 tensor 需要长度一致，所以要补齐 padding
                pad_len = attention_mask.shape[1] - valid_len
                # 用 'O' 标签的 ID 进行填充
                if pad_len > 0:
                    bio_ids += [tags["O"]] * pad_len
                preds_bio.append(bio_ids)
            # 将预测结果转换为 tensor
            preds = torch.tensor(preds_bio, device=device)

            # ------- 对齐：让预测和标签在送入评测函数前，格式完全一致-------
            for p_ids, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = int(mask.sum().item())
                # 只截取有效长度的部分，去掉 padding
                p_ids = [int(p) for p in p_ids[:valid_len]]
                l_ids = l_ids[:valid_len].tolist()
                # 遍历，去掉 [CLS], [SEP], 'X' (子词标记) 等不参与评测的特殊 token
                kept_pred, kept_gold = [], []
                for pid, lid in zip(p_ids, l_ids):
                    tag_name = idx2tag.get(lid, "O")
                    # 跳过特殊 token
                    if tag_name in ("[CLS]", "[SEP]", "X"):
                        continue
                    kept_pred.append(pid)
                    kept_gold.append(lid)
                # 将清理干净的预测和标签存入总列表
                all_preds.append(kept_pred)
                all_labels.append(kept_gold)
                all_words.append([])

    # 总体指标
    # 计算准确率、F1 值、精确率、召回率
    acc, f1, p, r = evaluate(all_preds, all_labels, all_words, tags)

    # 各类指标
    # 获取所有标签名称
    if isinstance(next(iter(tags.keys())), int):
        tag_names = list(tags.values())
    else:
        tag_names = list(tags.keys())
    # 提取实体类型（去掉 B-/I- 前缀）
    entity_types = sorted({name.split('-')[-1] for name in tag_names if '-' in name})
    # 分别计算每种实体类型的指标
    for ent_type in entity_types:
        f1_c, p_c, r_c = evaluate_each_class(all_preds, all_labels, all_words, tags, ent_type)
        print(f"[{ent_type}] P={p_c:.4f}, R={r_c:.4f}, F1={f1_c:.4f}")

    return acc, f1, p, r


def load_config(model_dir):
    """
    加载模型配置文件
    model_dir: 模型保存的目录名
    """
    # 拼接配置文件的完整路径
    config_path = os.path.join(script_dir, "save_models", model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    # 读取 JSON 配置文件
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    # 将字典转换为 Namespace 对象，方便通过属性访问
    return argparse.Namespace(**config_dict)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="保存模型name")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.save_name)
    # 覆盖配置中的 device 参数
    config.device = args.device
    device = torch.device(config.device)

    # 设置数据目录和图片目录
    data_dir = os.path.join(script_dir, 'data')
    img_path = os.path.join(data_dir, 'ner_img')

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 初始化数据处理器
    processor = DataProcessor(data_dir, config.text_encoder)

    # 测试集
    # 创建测试数据集实例
    test_dataset = NERDataset(
        processor, transform,
        img_path=img_path, max_seq=config.max_len,
        sample_ratio=1.0, mode='test'
    )

    # 创建 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 模型
    # 构建模型并移动到指定设备
    model = build_model(config).to(device)
    # 加载模型权重
    model_path = os.path.join(script_dir, "save_models", args.save_name, "model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # 评估
    # 在测试集上评估模型性能
    acc, f1, p, r = evaluate_model(model, test_loader, device, test_dataset.label_mapping)
    print(f"[Overall] Acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
