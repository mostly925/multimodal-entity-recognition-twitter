# -*- coding: utf-8 -*-

import codecs
import numpy as np


def get_chunks(seq, tags):
    """
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]每个元素是 (实体类型, 开始索引, 结束索引)
    """
    # 获取 "O" 标签对应的 ID
    default = tags['O']
    # 创建一个反向映射字典，从 ID 映射回标签名
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    # 初始化一个空列表，用于存放return的实体块。
    chunks = []
    # 初始化两个状态变量，用来追踪当前正在处理的实体块。
    # chunk_type: 当前实体块的类型 (如 "PER", "LOC")，None 表示当前不在任何实体块内。
    # chunk_start: 当前实体块的开始索引，None 表示当前不在任何实体块内。
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # 遇到 'O' 标签，并且我们正在一个实体块内部
        if tok == default and chunk_type is not None:
            # 创建一个元组来表示这个刚刚结束的实体块
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # 遇到了一个非 'O' 标签 (即 B-* 或 I-*) 
        elif tok != default:
            # 调用辅助函数 get_chunk_type 来解析当前标签
            # tok_chunk_class 判断是以B开头还是I开头
            # tok_chunk_type 判断是什么类型，PER,LOC
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            
            if chunk_type is None:# 意味着一个新实体开始了
                chunk_type, chunk_start = tok_chunk_type, i
            # 正在一个实体块内部，但遇到了一个新实体
            # 新实体有两种可能：1. 类型不同 (例如从 PER 切换到 LOC)   2. 类型相同但以 'B-' 开头
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                # 先把上一个实体块保存下来。结束位置是当前索引 i
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                # 然后，以当前位置为起点，开始记录这个新的实体块
                chunk_type, chunk_start = tok_chunk_type, i
        
        else:#遇到 'O' 标签，但我们本来就不在实体块内，或者遇到 I-* 标签，并且它与当前实体块类型一致 (这是最常见的情况，实体在延续)
            pass
    # 检查循环结束后是否还有一个未闭合的实体块 (例如实体在句子末尾结束)
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: 标签的 ID，比如整数 4。
        idx_to_tag: 一个从 ID 映射到标签名字符串的字典，例如 {4: "B-PER", ...}。
    Returns:
        一个元组，包含两个字符串，例如 ("B", "PER")。
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


# def run_evaluate(self, sess, test, tags):
# def run_evaluate(self, sess, test, tags):
def evaluate(labels_pred, labels, words, tags):
    """
    Args:
        labels_pred: 预测的标签ID序列列表, e.g., [[...], [...], ...]
        labels: 真实的标签ID序列列表, e.g., [[...], [...], ...]
        words: 原始单词序列列表 (在这个函数里实际没用到核心计算，主要是为了兼容旧接口)
        tags: 标签名到ID的映射字典
    Returns:
        acc: 准确率 (基于单个标签)
        f1: F1分数 (基于实体块)
        p: 精确率 (基于实体块)
        r: 召回率 (基于实体块)
    """

    

    index = 0
    sents_length = []
    # 初始化一个空列表，用于存放每个位置上标签是否预测正确的布尔值 (True/False)
    accs = []
    # 初始化三个计数器，用于计算 P, R, F1。它们都是浮点数以保证除法精度。
    # correct_preds: 预测正确并且真实存在的实体块数量 (TP, True Positives)
    # total_preds:   模型总共预测出的实体块数量 (TP + FP, False Positives)
    # total_correct: 数据集中总共真实存在的实体块数量 (TP + FN, False Negatives)
    correct_preds, total_correct, total_preds = 0., 0., 0.
    # 使用 zip 同时遍历预测标签、真实标签和单词（尽管单词没用）。
    # 每次循环处理一个句子。
    # lab: 当前句子的真实标签序列, e.g., [9, 9, 1, 2]
    # lab_pred: 当前句子的预测标签序列, e.g., [9, 9, 1, 9]
    # word_sent: 当前句子的单词序列
    for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
        word_st = word_sent
        lab = lab
        lab_pred = lab_pred
        accs += [a == b for (a, b) in zip(lab, lab_pred)]
        lab_chunks = set(get_chunks(lab, tags))
        lab_pred_chunks = set(get_chunks(lab_pred, tags))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)


    # 计算精确率 (Precision) = 预测正确的实体数 / 模型预测的总实体数
    p = correct_preds / total_preds if correct_preds > 0 else 0
    # 计算召回率 (Recall) = 预测正确的实体数 / 数据集中真实的总实体数
    r = correct_preds / total_correct if correct_preds > 0 else 0
    
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)


    return acc, f1, p, r


def evaluate_each_class(labels_pred, labels, words, tags, class_type):
    # class_type: 指定要评估的实体类型字符串，例如 "PER", "LOC", "ORG"
    index = 0

    accs = []
    # correct_preds_cla_type: 预测正确且类型为 class_type 的实体块数量 (TP for class_type)
    # total_preds_cla_type:   模型预测出的类型为 class_type 的实体块总数 (TP+FP for class_type)
    # total_correct_cla_type: 数据集中真实的类型为 class_type 的实体块总数 (TP+FN for class_type)
    correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.
    
    # 同样，使用 zip 同时遍历预测标签、真实标签和单词。
    # 每次循环处理一个句子。
    for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
        # 创建一个空列表，用来存放预测结果中类型为 xxx 的实体。
        lab_pre_class_type = []
        lab_class_type = []

        word_st = word_sent
        lab = lab
        lab_pred = lab_pred
        # --- 步骤 1: 提取所有实体块 ---
        # 从真实标签序列中提取所有类型的实体块。
        # lab_chunks: [('PER', 0, 2), ('LOC', 5, 6)]
        lab_chunks = get_chunks(lab, tags)
        # 从预测标签序列中提取所有类型的实体块。
        # lab_pred_chunks: [('PER', 0, 2), ('ORG', 8, 9)]
        lab_pred_chunks = get_chunks(lab_pred, tags)
        
        # --- 步骤 2: 筛选出我们关心的特定类型的实体块 ---       
        for i in range(len(lab_pred_chunks)):
            if lab_pred_chunks[i][0] == class_type:
                lab_pre_class_type.append(lab_pred_chunks[i])
        lab_pre_class_type_c = set(lab_pre_class_type)

        for i in range(len(lab_chunks)):
            if lab_chunks[i][0] == class_type:
                lab_class_type.append(lab_chunks[i])
        lab_class_type_c = set(lab_class_type)
        # --- 步骤 3: 基于筛选后的实体块进行计数 ---
        # 将所有真实的实体块也转为集合，用于后续交集运算。
        lab_chunksss = set(lab_chunks)
        correct_preds_cla_type += len(lab_pre_class_type_c & lab_chunksss)
        total_preds_cla_type += len(lab_pre_class_type_c)
        total_correct_cla_type += len(lab_class_type_c)

    p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
    r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0

    return f1, p, r


if __name__ == '__main__':
    max_sent = 10
    tags = {'0': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-LOC': 3, 'I-LOC': 4,
            'B-ORG': 5, 'I-ORG': 6,
            'B-OTHER': 7, 'I-OTHER': 8,
            'O': 9}
    labels_pred = [
        [9, 9, 9, 1, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 1, 3, 1, 2, 0, 0, 0]
    ]
    labels = [
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0]
    ]
    words = [
        [0, 0, 0, 0, 0, 3, 6, 8, 5, 7],
        [0, 0, 0, 4, 5, 6, 7, 9, 1, 7]
    ]
    id_to_vocb = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
    new_words = []
    for i in range(len(words)):
        sent = []
        for j in range(len(words[i])):
            sent.append(id_to_vocb[words[i][j]])
        new_words.append(sent)
    class_type = 'PER'
    acc, f1, p, r = evaluate(labels_pred, labels, new_words, tags)
    print(p, r, f1)
    f1, p, r = evaluate_each_class(labels_pred, labels, new_words, tags, class_type)
    print(p, r, f1)
