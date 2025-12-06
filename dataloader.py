# -*- coding: utf-8 -*-
import json
import logging
import os
import random
from typing import List, Dict
from transformers import BertConfig
from transformers import BertTokenizer
import PIL # 用于图像文件的读取和处理
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from transformers import RobertaTokenizer, CLIPProcessor
from model import _resolve_path



logger = logging.getLogger(__name__)

# ===== Span 类型映射（Twitter2015/2017：LOC/ORG/OTHER/PER）=====
# 定义一个全局字典，将实体类型字符串映射到一个固定的整数ID
TYPE2ID_SPAN = {"LOC": 0, "ORG": 1, "OTHER": 2, "PER": 3}

# 将传统的 BIO 标签序列转换成 (起始位置, 结束位置, 实体类型) 的格式
def bio_ids_to_spans(label_ids, id2tag, x_token_id: int):
    # id2tag:一个映射字典{id: tag_str}，如 {1:'O', 2:'B-MISC', ...}
    """
    将“未加 [CLS]/[SEP]”的 BIO 标签序列转为 spans（左闭右开）。
    - label_ids: 展开到子词后的标签id序列（子词续接位是 'X'）
    - 
    - x_token_id: label_mapping['X']
    返回: List[(start, end, type_str)]
    """
    # 初始化一个空列表，用于存放找到的实体 span
    spans = []
    # i 是当前遍历的位置，n 是序列总长
    i, n = 0, len(label_ids)
    while i < n:# 遍历整个标签序列
        lid = label_ids[i]# 获取当前位置的标签ID
        if lid == x_token_id:  # 如果是 'X' 标签 (代表这是一个词的非首个子词)，直接continue跳过，处理下一个
            i += 1
            continue
        # 将标签ID转换为字符串形式，如 "B-PER"
        tag = id2tag.get(lid, "O")
        # 如果标签以 "B-" 开头，说明一个新实体开始了
        if tag.startswith("B-"):
            
            # 提取实体类型，如 "PER"
            ent = tag.split("-")[1]
            
            # 如果遇到"MISC"，特殊处理，将 "MISC" 统一为 "OTHER"
            if ent == "MISC":
                ent = "OTHER"
                
            j = i + 1 # j 用于向后查找实体的结束位置
            # 只要索引 j 还没有越界，并且 j 位置的标签正好是当前实体类型对应的 "I-类型" 的标签，那么就继续这个循环
            # label_ids[j]获取索引 j 处的标签 ID，id2tag将 数字ID 转换回字符串标签
            while j < n and id2tag.get(label_ids[j], "O") == f"I-{ent}":
                j += 1
            # 循环结束后，[i, j) 就是一个完整的实体区间
            spans.append((i, j, ent))
            # 将 i 直接跳到实体结束后，继续寻找下一个实体
            i = j
        # 如果标签不以 "B-" 开头，跳到下一个位置
        else:
            i += 1
    return spans # 返回所有找到的实体 span 列表

# 读取和解析原始的 .txt 数据文件
class DataProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path# 数据文件所在的目录路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本所在目录
        
        # 根据传入的 bert_name (如 "roberta-base")，加载对应的 Tokenizer
        t_path = _resolve_path(self.script_dir, bert_name)# 查找模型路径：如果本地没有下载，t_path 就是 "roberta-base"字符串
        # 如果没有，.from_pretrained自动下载
        if bert_name == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained(t_path,do_lower_case=True)# do_lower_case=True在进行分词之前，将所有输入的文本都转换为小写字母，显著减少需要模型学习的词汇量
        elif bert_name == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(t_path,do_lower_case=True)

    def get_label_mapping(self):
        # 定义所有可能的标签，并创建一个从标签名到整数ID的映射
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]",
                      "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0# 指定 "PAD" 标签的ID为0
        return label_mapping
        
        
    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        从 JSONL 文件加载数据
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        # 根据模式('train'/'valid'/'test')确定要读取的文件
        # 映射你的文件名
        mode_map = {'train': 'train.json', 'valid': 'dev.json', 'test': 'test.json'}
        file_name = mode_map.get(mode)
        if not file_name:
            raise ValueError(f"Invalid mode: {mode}")
            
         # 注意：这里的 self.data_path 应该指向包含 train.json 等文件的目录，例如 "data/"
        load_file = os.path.join(self.script_dir, self.data_path, file_name)
        
        all_data = []
        with open(load_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                all_data.append(json.loads(line))
        # 从解析后的 JSON 对象中提取需要的信息
        # 你的 JSON 已经包含了 tokens, label, 和 image_id
        raw_words = [item['tokens'] for item in all_data]
        raw_targets = [item['label'] for item in all_data]
        
        # 对 None 的判断
        imgs = []
        for item in all_data:
            iid = item.get('image_id')
            if iid is not None:
                imgs.append(str(iid) + '.jpg')# 假设图片都是.jpg格式   
            else:
                # 给一个不存在的文件名，触发 Dataset.__getitem__ 里的 try-except 逻辑
                # 从而自动加载 data/no_images.jpg
                imgs.append("no_image_placeholder") 
        # 确保解析出的句子、标签、图片数量一致
        assert len(raw_words) == len(raw_targets) == len(imgs)
        

        # 如果需要采样（比如在资源有限时做快速实验）
        # 如果 sample_ratio 是 1.0，意味着使用100%的数据，那么就不需要执行采样逻辑
        # 只有当 sample_ratio 小于 1.0 时，才会执行下面的采样操作
        if sample_ratio != 1.0:
            # 生成所有样本的索引列表，计算需要采样的样本数量 k，random.choices表示有放回地从所有样本的索引列表随机抽出k个，k=所有句子数 * 采样比
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
            # 从原始句子、标签、图片中取出刚才采样的样本
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            # 检查采样后的三个新列表的长度是否完全相等
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(
                len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs}
        
        # 返回一个包含数据集所有句子、标签、图片的字典
        return {"words": raw_words, "targets": raw_targets, "imgs": imgs}

    

# 继承了 PyTorch 的 Dataset。核心功能是处理单个数据样本（通过 __getitem__ 方法）
class NERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40, sample_ratio=1, mode='train',
                 ignore_idx=0, return_span: bool=False) -> None:
        self.processor = processor# 传入上面定义的处理器实例
        self.transform = transform# 传入图像预处理的变换（如缩放、归一化）
        self.data_dict = processor.load_from_file(mode, sample_ratio) # 调用处理器加载数据
        self.tokenizer = processor.tokenizer#  获取 Tokenizer
        self.label_mapping = processor.get_label_mapping() # 获取标签映射
        self.id2tag = {v: k for k, v in self.label_mapping.items()}# 创建一个ID到标签名的映射
        self.max_seq = max_seq # 句子的最大长度
        self.ignore_idx = ignore_idx # padding 位置的标签ID
        self.img_path = img_path# 图像文件所在的根目录
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.return_span = return_span# 设置一个开关，决定 __getitem__ 方法在处理完一个样本后，最终返回哪种格式的数据（传统的 BIO 标签序列/实体 Span）。

        # 缓存常用的标签ID，提高效率
        self.cls_id = self.label_mapping.get("[CLS]")
        self.sep_id = self.label_mapping.get("[SEP]")
        self.x_id   = self.label_mapping.get("X")

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        # 根据索引 idx 获取并处理第 idx 个样本
        word_list, label_list, img = (self.data_dict['words'][idx],
                                      self.data_dict['targets'][idx],
                                      self.data_dict['imgs'][idx])

        # 词→子词展开，并同步展开标签（非首子词置 'X'）
        tokens, labels_exp = [], []
        # 循环将word_list展开成每个单词的形式
        for i, word in enumerate(word_list):
            sub = self.tokenizer.tokenize(word)# 将一个单词（如 "playing"）分词（如 ["play", "##ing"]）
            if not sub:# 如果分词结果为空（罕见情况）
                sub = [self.tokenizer.unk_token]# 使用未知词标记
            tokens.extend(sub) # 将子词列表加入总的 tokens 列表
            lab = label_list[i]# 当前正在处理的单词 word 对应的原始标签字符串
            # 与你原逻辑保持一致：OTHER → MISC
            if 'OTHER' in lab:
                lab = lab[:2] + 'MISC'
            # 将标签字符串映射到数字 ID
            lab_id = self.label_mapping.get(lab, self.label_mapping["O"])
            # 一个原始单词（Washington）可能被分成多个子词（['Washing', '##ton']）。但是我们只有一个标签（B-LOC）。那么这两个子词应该分别对应什么标签呢？
            # 处理子词 (subword) 标签对齐：
            # ①单词的第一个子词继承这个单词的完整标签。
            # ②这个单词的所有后续子词，用 'X' 来表示标签。这个 'X' 标签在后续计算损失或评估时通常会被忽略，它仅仅是一个占位符，表示“这里是前一个子词的延续”。
            for m in range(len(sub)):
                labels_exp.append(lab_id if m == 0 else self.label_mapping["X"])

        # 截断（给 [CLS]/[SEP] 留出2位）
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[: self.max_seq - 2]# 确保处理后的单词序列（tokens）有足够的空间来添加特殊的起始符 [CLS] 和结束符 [SEP]，同时保证总长度不超过模型设定的最大长度 self.max_seq
            labels_exp = labels_exp[: self.max_seq - 2]

        # 将一个 BIO 格式的标签序列（即 labels_exp）转换成一个实体Span列表
        spans = bio_ids_to_spans(labels_exp, self.id2tag, x_token_id=self.x_id)

        # 编码（加 [CLS]/[SEP],填充标记 [PAD]）
        encode_dict = self.tokenizer.encode_plus(
            tokens, max_length=self.max_seq, truncation=True, padding='max_length'
        )
        # 返回
        # {
        # "input_ids": [101, 2023, 2003, ..., 102, 0, 0, 0],  // 长度为 max_seq
        # "attention_mask": [1, 1, 1, ..., 1, 0, 0, 0],     // 长度为 max_seq
        # "token_type_ids": [0, 0, 0, ..., 0, 0, 0, 0]      // 用于句子对任务，这里我们用不到
        # }
        input_ids = torch.tensor(encode_dict['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encode_dict['attention_mask'], dtype=torch.long)

        # 构建一个与 input_ids 长度完全一致、且位置对齐的标签序列
        # input_ids = [CLS_ID] + token_IDs + [SEP_ID] + [PAD_ID] * N
        labels = [self.label_mapping["[CLS]"]] + labels_exp + [self.label_mapping["[SEP]"]] + [self.ignore_idx] * (self.max_seq - len(labels_exp) - 2)
        labels = torch.tensor(labels, dtype=torch.long)

        # 因为前面加了一个 [CLS]，所以需要将 spans 整体 +1 偏移
        shifted_spans = []
        for s, e, t_str in spans:
            ss, ee = s + 1, e + 1
            # 【边界检查】：确保偏移后的区间仍然在有效范围内
            if 0 <= ss < ee <= self.max_seq - 1:
            # 【类型转换】：将人类可读的标签 "LOC" 转换为模型能理解的数字 ID，TYPE2ID_SPAN 是一个全局字典：{"LOC": 0, "ORG": 1, ...}
                t_id = TYPE2ID_SPAN.get(t_str, None)
                if t_id is not None:
                    shifted_spans.append((ss, ee, t_id))

        # 图像读取
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)# 对图片进行预处理
            except:
                # 如果上述步骤出错（例如文件找不到、损坏）
                # 构造一张默认图片的路径
                img_path = os.path.join(self.script_dir, 'data', 'no_images.jpg')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            
            # 如果需要 span 格式
            if self.return_span:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "image": image,
                    "spans": shifted_spans  # List[(s,e,type_id)]
                }
            # 如果不需要 span 格式
            else:
                return input_ids, attention_mask, labels, image

        # 无图像分支
        if self.return_span:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "spans": shifted_spans
            }
        else:
            assert len(input_ids) == len(attention_mask) == len(labels)
            return input_ids, attention_mask, labels

def collate_fn_span(batch):
    """
    batch:包含了 DataLoader 从 Dataset 中取出的一批样本
      input_ids:[T], attention_mask:[T], labels:[T], image: CxHxW(可选), spans: List[(s,e,type)]
    返回：
      - input_ids:   [B,T]
      - attention_mask:[B,T]
      - labels:      [B,T]   （token标签，兼容多任务或评测）
      - images:      [B,C,H,W]（若有）
      - span_starts / span_ends / span_types: [B, S_max]（-1 填充）
      - span_mask:   [B, S_max]（1/0）
      - span_counts: [B]      每条样本 span 数
    """
    # 检查这批样本中是否包含图像数据：只需要看第一个样本（batch[0]）就行，因为同一批次的数据结构都是一样的。
    has_image = ("image" in batch[0])
    # 同一batch的input_ids、attention_mask、labels、images堆叠起来
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    if has_image:
        images = torch.stack([b["image"] for b in batch], dim=0)
    # --- 处理形状不固定的数据 (spans) ---
    spans_list = [b["spans"] for b in batch]# 取出批次中每个样本的spans列表
    counts = [len(s) for s in spans_list]
    Smax = max(counts) if counts else 0  # 找到这个批次中最大的span
    if Smax == 0:
        Smax = 1  # 保底一列，避免下游维度为0
    # torch.full 创建一个指定形状的张量，并用一个值（-1）来填充它。
    # -1 是一个常用的填充值，因为在计算损失时，PyTorch的损失函数可以设置 ignore_index=-1 来忽略这些位置。
    span_starts = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_ends   = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_types  = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_mask   = torch.zeros((len(batch), Smax), dtype=torch.long)
    # 遍历每个样本，将真实的span数据填入“大盒子”中
    # i 是样本在批次中的索引 (0, 1, 2, ...)，spans 是该样本的span列表。
    for i, spans in enumerate(spans_list):
        # j 是span在当前样本中的索引，(s,e,t)是span的具体内容
        for j, (s,e,t) in enumerate(spans[:Smax]):# [:Smax] 是个安全措施，不足Smax不会被截断。
            span_starts[i, j] = s
            span_ends[i, j]   = e
            span_types[i, j]  = t
            span_mask[i, j]   = 1 # 在mask的对应位置标记为1，表示这是真实数据

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "span_starts": span_starts,
        "span_ends": span_ends,
        "span_types": span_types,
        "span_mask": span_mask,
        "span_counts": torch.tensor(counts, dtype=torch.long),
    }
    if has_image:
        out["image"] = images
    return out


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    DATA_PATH = {
        "twitter2015": {
            # text data
            'train': 'data/twitter2015/train.txt',
            'valid': 'data/twitter2015/valid.txt',
            'test': 'data/twitter2015/test.txt',
        },
        "twitter2017": {
            # text data
            'train': 'data/twitter2017/train.txt',
            'valid': 'data/twitter2017/valid.txt',
            'test': 'data/twitter2017/test.txt',
        }
    }
    # image data
    IMG_PATH = {
        'twitter15': 'data/twitter2015/twitter2015_images',
        'twitter17': 'data/twitter2017/twitter2017_images',
    }
    # 将任意尺寸的原始输入图像转换成一个尺寸固定、数值范围标准化的Tensor，以便输入到深度学习模型（尤其是像 CLIP、ResNet 等在 ImageNet 上预训练过的模型）中
    transform = transforms.Compose([
        transforms.Resize(256),# 调整图像尺寸：将图像的较短边缩放到 256 像素，另一条边则按原始图像的宽高比进行等比例缩放。举例一张 400 x 300 的图像，较短边是 300。它会被缩放成 341 x 256 (因为 400 * (256/300) ≈ 341)
        transforms.CenterCrop(224),# 从中心进行裁剪：将上一步裁剪的图片从中心位置裁剪出一个 224 x 224 像素的正方形区域
        transforms.ToTensor(),# 将图像数据转换为 PyTorch 张量 (Tensor)
        # 对图像张量进行标准化：对输入的 C x H x W 张量的每个通道均值和标准差进行统一
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    data_path = DATA_PATH['twitter2015']
    img_path = IMG_PATH['twitter15']
    processor = DataProcessorr(data_path, "chinese-roberta-www-ext")
    train_dataset = NERDataset(processor, transform, img_path=img_path, max_seq=128,
                                  sample_ratio=1.0, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    for batch in train_dataloader:
        input_ids, token_type_ids, attention_mask, labels, image = batch

        print(input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape)
        break
