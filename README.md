# 多模态命名实体识别 (MNER) 项目运行手册

## 📋 项目概述

本项目实现了一个基于 **RoBERT + CLIP** 的多模态命名实体识别系统，支持从文本和图像中联合提取命名实体。

---

## 🚀 快速开始

### 1. 环境安装

#### 必需依赖
```bash
pip install -r requirements.txt
```

核心依赖包括：
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `torchvision`
- `pytorch-crf`
- `wandb`（可选，用于实验追踪）

#### 预训练模型下载

项目需要以下预训练模型（首次运行时会自动下载，建议提前下载到本地）：

**方式一：自动下载**（推荐）
- 直接运行训练脚本，`transformers` 库会自动从 Hugging Face 下载模型

**方式二：手动下载**（国内网络不稳定时推荐）
1. **文本编码器**：`roberta-base`

   ```bash
   # 从 Hugging Face 下载并放置到项目根目录
   mkdir roberta-base
   # 将 config.json, pytorch_model.bin, vocab.json, merges.txt 等文件放入
   ```

2. **视觉编码器**：`openai/clip-vit-base-patch32`
   ```bash
   mkdir -p openai/clip-vit-base-patch32
   # 将 CLIP 模型文件放入此目录
   ```

代码会优先查找本地目录，找不到时才联网下载。

---

### 2. 数据准备

#### 数据格式

数据应放置在 `data/` 目录下，包含：
- `train.json`：训练集
- `dev.json`：验证集
- `test.json`：测试集
- `ner_img/`：图像文件夹

#### JSONL 格式示例
每行一个 JSON 对象：

```json
{
  "id": 0,
  "image_id": "12345",
  "text": "Apple is buying U.K. startup",
  "tokens": ["Apple", "is", "buying", "U.K.", "startup"],
  "label": ["B-ORG", "O", "O", "B-LOC", "O"],
  
}
{
  "id": 1,
  "image_id": "55555",
  "text": "我爱天安门",
  "tokens": ["我", "爱", "天", "安.", "门"],
  "label": ["O", "O", "B-LOC", "I-LOC", "I-LOC"],
  
}
```

- `tokens`：分词后的单词列表
- `label`：BIO 标注法:
  - **"O"**：非实体/标点
  - **"B-PER" / "I-PER"**： Person (人名)
  - **"B-ORG" /  "I-ORG"**：Organization (组织/机构/公司)
  - **"B-LOC" / "I-LOC"**：Location (地点/地名)
  - **"B-MISC" / "I-MISC"**：Miscellaneous (杂项/其他)
- `image_id`：图像文件名（不含扩展名），对应 `data/ner_img/{image_id}.jpg`

---

## 🏋️ 模型训练

### 基础训练命令

```bash
python train.py \
    --model MNER \
    --text_encoder roberta-base \
    --image_encoder openai/clip-vit-base-patch32 \
    --use_image \
    --use_bilstm \
    --epochs 100 \
    --batch_size 32 \
    --max_len 128 \
    --device cuda:0
```

### 启用 WandB 监控

```bash
python train.py --use_wandb --wandb_project "MyNER" 
```

首次使用需登录 WandB：
```bash
wandb login
# 输入 API Key（从 https://wandb.ai/authorize 获取）
```

### 高级训练选项

#### 1. 对抗训练（FGM）
```bash
python train.py --use_fgm --fgm_epsilon 1.0
```

#### 2. 混合精度训练（加速 + 省显存）
```bash
python train.py --use_amp
```

#### 3. 微调视觉编码器
```bash
python train.py --vision_trainable --unfreeze_last_vision_blocks 2
```

#### 4. 断点续训
```bash
python train.py --continue_train_name "2025-12-03_train_MNER_experiment1"
```

### 完整训练示例

```bash
python train.py \
    --model MNER \
    --use_image \
    --use_bilstm \
    --epochs 50 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --fin_tuning_lr 5e-5 \
    --clip_lr 1e-5 \
    --downs_en_lr 3e-4 \
    --use_fgm \
    --use_amp \
    --use_wandb \
    --wandb_project "MNER-Twitter" \
    --ex_name "exp1_fgm_amp"
```

---

## 模型评估

### 在测试集上评估

```bash
python test.py \
    --save_name "2025-12-03_train_MNER_exp1_fgm_amp" \
    --device cuda:0
```

**输出示例**：
```
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[LOC] P=0.0096, R=0.0224, F1=0.0134
[MISC] P=0.0124, R=0.0298, F1=0.0175
[ORG] P=0.0817, R=0.1018, F1=0.0907
[PER] P=0.0000, R=0.0000, F1=0.0000
[Overall] Acc=0.5352, P=0.0219, R=0.0314, F1=0.0258
```

## 📦 导出 ONNX 模型

用于加速推理部署：

```bash
python export_onnx.py --save_name 保存的文件名
```

生成文件：`mnre_model.onnx`

推理：

```python
python inference_onnx.py
```

---

## ⚙️ 配置参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `MNER` | 模型类型 |
| `--text_encoder` | `roberta-base` | 文本编码器 |
| `--image_encoder` | `openai/clip-vit-base-patch32` | 图像编码器 |
| `--use_image` | `False` | 是否使用图像模态（flag） |
| `--use_bilstm` | `False` | 是否使用 BiLSTM（flag） |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `128` | 批次大小 |
| `--max_len` | `128` | 最大序列长度 |
| `--device` | `cuda:0` | 计算设备 |

### 学习率（差分学习率）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fin_tuning_lr` | `5e-5` | RoBERTa 学习率 |
| `--clip_lr` | `1e-5` | CLIP 学习率（若启用训练） |
| `--downs_en_lr` | `3e-4` | 下游任务层学习率 |

### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--align_lambda` | `0.2` | 对齐损失权重 |
| `--preserve_lambda` | `0.05` | 保真损失权重 |
| `--nce_lambda` | `0.02` | InfoNCE 损失权重 |
| `--sparsity_lambda` | `0.01` | 稀疏性损失权重 |
| `--lambda_type` | `1.0` | Span 类型分类损失权重 |
| `--aux_ce_lambda` | `0.5` | 辅助 Token 分类损失权重 |

### 训练技巧

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_fgm` | `False` | 启用 FGM 对抗训练（flag） |
| `--fgm_epsilon` | `1.0` | FGM 扰动系数 |
| `--use_amp` | `False` | 启用混合精度训练（flag） |
| `--gradient_accumulation_steps` | `2` | 梯度累积步数 |
| `--clip_grad` | `2.0` | 梯度裁剪阈值 |
| `--warmup_prop` | `0.1` | 学习率预热比例 |
| `--image_dropout_p` | `0.5` | 图像 Dropout 概率 |

### WandB 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_wandb` | `False` | 启用 WandB（flag） |
| `--wandb_project` | `MNER` | WandB 项目名称 |
| `--wandb_entity` | `None` | WandB 用户/组织名 |

### 早停策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min_epoch_num` | `5` | 最小训练轮数 |
| `--patience` | `0.00001` | F1 提升阈值 |
| `--patience_num` | `20` | 早停耐心值 |

---

## 📁 项目结构

```
NER/
├── config.py              # 配置参数定义
├── train.py               # 训练脚本
├── test.py                # 评估脚本
├── model.py               # 模型定义（MultimodalNER 等）
├── dataloader.py          # 数据加载器
├── metrics.py             # 评估指标
├── visualize.py           # 注意力可视化
├── requirements.txt       # 依赖列表
├── data/                  # 数据目录
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── ner_img/          # 图像文件夹
│   └── no_images.jpg     # 占位图
├── onnx/  
│   ├── export_onnx.py    # ONNX 导出
│   ├── inference_onnx.py # ONNX 推导			
├── save_models/          # 模型保存目录（训练后生成）
└── README.md             # 本文档
```

---

## 🛠️ 常见问题

### Q1: 模型下载失败怎么办？
**A**: 手动下载模型到本地：

1. 访问 [Hugging Face](https://huggingface.co/)
2. 下载 `roberta-base` 和 `openai/clip-vit-base-patch32`
3. 放置到项目根目录对应文件夹

### Q2: 显存不足（OOM）怎么办？
**A**: 尝试以下方案：
- 减小 `--batch_size`（如 16 或 8）
- 减小 `--max_len`（如 64）
- 启用 `--use_amp` 混合精度
- 增加 `--gradient_accumulation_steps`

### Q3: 训练速度慢？
**A**: 
- 启用 `--use_amp` 混合精度加速
- 设置 `num_workers=4` 并启用 `pin_memory=True`（已默认）
- 不启用 `--vision_trainable`（冻结 CLIP）

### Q4: 如何使用纯文本模式？
**A**: 不传 `--use_image` 参数即可：
```bash
python train.py --model MNER  # 不加 --use_image
```

### Q5: WandB 登录问题？
**A**: 
```bash
# 离线模式
export WANDB_MODE=offline
python train.py --use_wandb

# 或在代码中设置 mode="offline"（已默认）
```
