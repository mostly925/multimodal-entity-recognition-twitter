# -*- coding: utf-8 -*-

import os
import json
import json5
import re

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_conll_to_json(file_path):
    # 初始化一个空列表，用来存放所有处理好的样本数据。
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        # 读取文件中的所有行，并将它们作为一个字符串列表存储在变量 lines 中。
        lines = f.readlines()
    # 初始化一个变量 img_id 为 None（空值）。存储当前正在处理的样本的图片ID。
    img_id = None
    # tokens 用来暂存一个样本中的所有单词。
    # labels 用来暂存一个样本中与单词一一对应的所有标签。
    tokens, labels = [], []
    # 开始遍历文件中的每一行。
    # 这里的 lines + [''] 是一个很巧妙的技巧：在所有行的末尾追加一个空字符串。确保文件中的最后一个样本也能被正确处理，因为处理逻辑是在遇到下一个样本标记或文件末尾时触发的。
    for line in lines + ['']: 
        line = line.strip()
        if line.startswith("IMGID:"):# 检查当前行是否以 "IMGID:" 开头
            if img_id:
                # 调用 format_one 函数，将收集到的 img_id, tokens, labels 格式化成一个字典。
                # 然后将这个字典追加到最终的 data 列表中
                data.append(format_one(img_id, tokens, labels))
                tokens, labels = [], []
            # 将当前行的内容（即新的图片ID，如 "IMGID:123.jpg"）赋值给 img_id
            img_id = line
        elif line == '':# 如果行是空的（通常用作分隔符），则跳过这一行，继续下一次循环
            continue
        else:# 如果行既不是 IMGID 也不是空行，那么它就是 "单词 标签" 格式的数据行
            parts = line.split()
            # 确保分割后至少有两个部分（一个单词和一个标签）
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels.append(parts[1])
    # 这个 if 语句是在 for 循环结束之后执行的。
    # 它的作用是处理文件中的最后一个样本。因为循环结束后，最后一个样本的数据还在 tokens 和 labels 里，没有被处理。
    # 条件 `img_id and tokens` 确保只有在确实有数据时才执行。
    if img_id and tokens:
        data.append(format_one(img_id, tokens, labels))  # 最后一个样本

    return data


def format_one(img_id, tokens, labels):
    # img_id: 字符串，表示图片的ID
    # tokens: 列表，包含一个样本的所有单词，例如 ['Hello', 'world']
    # labels: 列表，包含与 tokens 一一对应的标签，例如 ['O', 'O']
    
    # 使用空格将 tokens 列表中的所有单词拼接成一个完整的字符串（句子）
    content = ' '.join(tokens)
    # 存放所有从句子中找到的实体信息
    entities = []
    i = 0
    # 只要索引 i 小于 labels 列表的长度，就一直执行。
    # 使用 while 循环而不是 for 循环，是因为在找到一个实体后，我们需要一次性跳过多个索引
    while i < len(labels):
        # 获取当前索引 i 对应的标签
        label = labels[i]
        # 检查当前标签是否以 'B-' 开头，标记着一个新实体的开始
        if label.startswith('B-'):
            # 如果是 'B-' 开头，提取实体类型。
            # label[2:] 表示从字符串的第3个字符开始截取到末尾。例如 'B-ORG' -> 'ORG'
            ent_type = label[2:]
            # 计算这个实体在 content 字符串中的起始位置（start index）。
            # ' '.join(tokens[:i]) 计算出当前单词之前所有单词拼接成的字符串的长度。
            # (1 if i > 0 else 0) 的作用是：如果不是第一个单词 (i > 0)，需要额外加 1，因为单词之间有一个空格
            start = len(' '.join(tokens[:i])) + (1 if i > 0 else 0)
            # 初始化一个结束位置的索引 end_i，从当前位置的下一个开始 (i + 1)
            end_i = i + 1
            # 循环条件是：
            # 1. end_i 没有超出 labels 列表的范围。
            # 2. 并且 end_i 对应的标签是以 'I-' (Inside) 开头的。'I-' 表示实体内部的词。
            while end_i < len(labels) and labels[end_i].startswith('I-'):
                # 如果满足条件，就将 end_i 加 1，继续向后检查
                end_i += 1
            # 当内部循环结束时，end_i 就指向了实体结束位置的下一个单词的索引。
            # 计算这个实体在 content 字符串中的结束位置（end index）。
            # ' '.join(tokens[:end_i]) 计算从开头到实体结束的所有单词拼接成的字符串的长度。
            end = len(' '.join(tokens[:end_i]))
            # 从原始 tokens 列表中，提取出这个实体的所有单词，并用空格拼接成字符串。
            # 例如，tokens[i:end_i] 会得到 ['Apple', 'Inc.']，拼接后就是 'Apple Inc.'。
            text = ' '.join(tokens[i:end_i])
            # 将这个找到的实体的信息（起始位置、结束位置、类型、文本）作为一个字典，追加到 entities 列表中
            entities.append({
                "start": start,
                "end": end,
                "type": ent_type,
                "text": text
            })
            # 下次循环将从这个实体之后开始
            i = end_i
        else:
            i += 1
    # 当循环结束后，返回一个组装好的字典
    return {
        "image": img_id,# 传入的图片ID
        "content": content,# 完整句子
        "entities": entities# 包含所有找到的实体的列表
    }


def convert_bio_block_to_json(block, image_base_dir):
    # block: 一个字符串，代表一个完整的数据样本块，通常由多行组成。
    # image_base_dir: 字符串，表示存放图片的基础目录路径，例如 "data/images"
    lines = block.strip().split("\n")
    # 确保处理的 block 不是空的，并且第一行必须以 "IMGID:" 开头。
    # 如果不满足这些条件，说明这个 block 格式不正确，函数直接返回 None，表示处理失败
    if not lines or not lines[0].startswith("IMGID:"):
        return None
    
    # 从第一行中提取图片ID。
    # lines[0].split(":") 会把 "IMGID:12345" 分割成 ['IMGID', '12345']。
    # [1] 取出第二个元素 '12345'。
    # .strip() 去除可能存在的多余空格。
    img_id = lines[0].split(":")[1].strip()
    # 使用 f-string 格式化字符串，拼接出完整的图片文件路径。
    # 例如，如果 image_base_dir 是 "data/twitter2015_images"，img_id 是 "123"，那么 image_path 就是 "data/twitter2015_images/123.jpg"
    image_path = f"{image_base_dir}/{img_id}.jpg"

    tokens = []
    labels = []
    # lines[1:] 表示从列表的第二个元素开始切片到末尾（因为第一行是 IMGID，已经处理过了）
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split()

        # ✅ 跳过 http 或 https 开头的网址
        # if len(parts) >= 1 and parts[0].startswith("http"):
        #     continue
        
        # 判断分割后的部分数量，以处理不同的行格式。
        # 如果只有一个部分，说明这一行只有一个单词，没有标签。
        if len(parts) == 1:
            # 那么，token 就是这个单词
            token = parts[0]
            # 标签被默认设置为 "O" (Outside)，表示这个词不属于任何实体
            label = "O"
        elif len(parts) == 2:
            # 如果正好有两个部分，这是最常见的情况，即 "单词 标签"
            token, label = parts
        else:
            raise ValueError(f"非法行格式: {line}")

        tokens.append(token)
        labels.append(label)

    # 对英文加空格，对中文不加
    if all(token.isascii() for token in tokens):
        text = " ".join(tokens)
    else:
        text = "".join(tokens)

    # print("\n")
    # ✅ 额外检查：tokens 和 labels 长度是否一致
    assert len(tokens) == len(labels), f"Token 和 Label 数量不一致：{tokens}, {labels}"

    return {
        "text": text,# 拼接好的完整文本
        "image_path": image_path,# 完整的图片路径
        "labels": labels# 包含所有标签的列表
    }


def convert_bio_txt_to_jsonl(input_txt_path, output_jsonl_path, image_base_dir="data/images"):
    # input_txt_path: 字符串，输入 .txt 文件的路径。
    # output_jsonl_path: 字符串，输出 .jsonl 文件的路径。
    # image_base_dir: 字符串，图片存放的基础目录，默认值为 "data/images"
    ''''txt文件的内容：
    IMGID:1001
    I	O
    love	O
    Beijing	B-LOC
    .	O'''
    
    with open(os.path.join(script_dir, input_txt_path), "r", encoding="utf-8") as f:
        content = f.read()
    
    # .split("\nIMGID:") 使用换行符加上 "IMGID:" 作为分隔符，将整个文件内容切分成多个块（blocks）。
    # 这样，每个 block 就代表一个数据样本（除了第一个 block 可能不包含 "IMGID:"）
    blocks = content.strip().split("\nIMGID:")
    '''[
    # 第一个块 (block 0)
    'IMGID:1001\nI\tO\nlove\tO\nBeijing\tB-LOC\n.\tO\n',
  
    # 第二个块 (block 1) - 注意，前面的 "\nIMGID:" 被作为分隔符切掉了
    '1002\nApple\tB-ORG\nInc.\tI-ORG\nis\tO\na\tO\ngreat\tO\ncompany\tO\n.\tO\n',
  
    # 第三个块 (block 2) - 同上
    '1003\nHe\tO\nis\tO\nJack\tB-PER\nMa\tI-PER\n.\tO'
    ]'''
    
    # 存放所有处理成功后的样本字典
    results = []
    for idx, block in enumerate(blocks):
        # 如果一个 block 在去除空白后是空的，就跳过它，继续处理下一个
        if not block.strip():
            continue
        # 如果当前不是第一个 block (idx != 0)，我们需要手动把 "IMGID:" 加回到 block 的开头，以还原其原始格式
        if idx != 0:
            block = "IMGID:" + block
        
        sample = convert_bio_block_to_json(block, image_base_dir)
        # 如果 sample 不是 None（即 block 被成功处理了）
        if sample:
            results.append(sample)

    # 写入 JSONL 文件
    with open(os.path.join(script_dir, output_jsonl_path), "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成，共处理 {len(results)} 条样本，输出至：{output_jsonl_path}")


import json
from collections import defaultdict


def convert_and_merge_by_img(input_file, output_file, image_prefix="twitter2017/twitter2017_images/"):
    # input_file: 字符串，输入文件的路径
    # output_file: 字符串，输出 .jsonl 文件的路径
    # image_prefix: 字符串，用于拼接图片完整路径的前缀，默认值为 "twitter2017/twitter2017_images/"
    grouped = defaultdict(list)
    # Step 1: 读入并按 img_id 分组
    with open(os.path.join(script_dir, input_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:# 如果是空行，则跳过
                continue
            data = json5.loads(line)
            # data["img_id"] 取出当前行数据对应的图片ID。
            # grouped[data["img_id"]] 会根据图片ID找到或创建一个列表。
            # .append(data) 将当前这行解析出的整个 data 字典，添加到对应图片ID的列表中
            grouped[data["img_id"]].append(data)

    # Step 2: 每个 img_id 合并处理
    merged_results = []

    for img_id, items in grouped.items():
        merged_tokens = []
        merged_labels = []

        for data in items:
            # 从当前数据项中提取出 'token' 列表
            tokens = data["token"]
            # 提取出头实体 'h' 的位置信息 'pos'
            h_pos = data["h"]["pos"]# [start_index, end_index] 
            # 提取出关系 'relation' 字符串
            relation = data["relation"]



            # 从关系字符串中提取出实体类型
            head_label = relation.strip("/").split("/")[0].upper()
            # 创建一个和当前 tokens 列表一样长，且所有元素都是 "O" 的标签列表
            labels = ["O"] * len(tokens)
            # 检查 h_pos 的起止位置是否有效（在 tokens 列表的范围内）
            if 0 <= h_pos[0] < h_pos[1] <= len(tokens):
                # 如果有效，将实体开始位置的标签设为 "B-" + 类型，例如 "B-PERSON"
                labels[h_pos[0]] = f"B-{head_label}"
                # 遍历从开始位置的下一个到结束位置之前的所有位置
                for i in range(h_pos[0] + 1, h_pos[1]):
                    # 将这些位置的标签设为 "I-" + 类型，例如 "I-PERSON"
                    labels[i] = f"I-{head_label}"

            merged_tokens.extend(tokens)
            merged_labels.extend(labels)

        merged_results.append({
            "text": " ".join(merged_tokens),
            "image_path": f"{image_prefix}{img_id}",
            "labels": merged_labels
        })

    # Step 3: 写出为 JSONL
    with open(os.path.join(script_dir, output_file), "w", encoding="utf-8") as fout:
        for item in merged_results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


# 使用方法
# data = parse_conll_to_json("yourfile.txt")
# import json; print(json.dumps(data, indent=2, ensure_ascii=False))


class DataProcessor:
    def __init__(self):
        # 当前脚本位置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def read_jsonl(file_path):
        """
        按行读取 json，每一行解析为一个字典对象，返回列表。
        """
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                data.append(json.loads(line))
        return data

    @staticmethod
    def save_jsonl(file_path, data, data_type):
        """
        将数据保存为 jsonl 格式，每行一个 JSON 对象。

        :param file_path: 保存文件路径
        :param data: list[dict]，每个元素是一条要保存的数据
        :param data_type: 字符串，如 "train", "test"，用于构成文件名
        """
        with open(os.path.join(file_path, f"{data_type}.jsonl"), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_twitter2015(self, dataset, data_type):
        file = os.path.join(self.script_dir, dataset, f"{data_type}.txt")
        if os.path.isfile(file):
            data = parse_conll_to_json(file)
            self.save_jsonl(os.path.join(self.script_dir, dataset), data, data_type)

    def process_twitter(self, dataset):
        # 处理训练集 (train.txt -> train.jsonl)
        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/train.txt",
            output_jsonl_path=f"{dataset}/train.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images"
        )
        # 处理测试集 (test.txt -> test.jsonl)
        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/test.txt",
            output_jsonl_path=f"{dataset}/test.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images"
        )
        # 处理验证集 (valid.txt -> valid.jsonl)
        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/valid.txt",
            output_jsonl_path=f"{dataset}/valid.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images"
        )

    def process_MORE(self, dataset="MNRE"):
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_train.txt",
            output_file=f"MNRE/train.jsonl",
            image_prefix="MNRE/mnre_image/train"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_val.txt",
            output_file=f"MNRE/valid.jsonl",
            image_prefix="MNRE/mnre_image/val"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_test.txt",
            output_file=f"MNRE/test.jsonl",
            image_prefix="MNRE/mnre_image/test"
        )

    def process(self, dataset):
        if dataset == 'twitter2015' or dataset == 'twitter2017':
            return self.process_twitter(dataset)
        if dataset == 'MNRE':
            return self.process_MORE(dataset)


if __name__ == '__main__':
    processor = DataProcessor()
    processor.process(dataset='twitter2015')
    # processor.process(dataset='twitter2017')
