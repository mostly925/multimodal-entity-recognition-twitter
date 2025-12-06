# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel

# =========================
#  模型注册表 & 工厂函数
# =========================
MODEL_REGISTRY: Dict[str, Callable[[object], nn.Module]] = {}
# 全局字典，用于存储模型名称和其对应的构造函数。
# 键 (str): 模型的名字，例如 "MNER"。
# 值 (Callable): 创建该模型实例的函数或类，例如 MultimodalNER 类。

# 构造一个装饰器：把模型类/构造函数注册进字典
def register_model(name: str):
    
    def deco(cls_or_fn):
        # 检查模型名字是否已经被注册，防止冲突。
        if name in MODEL_REGISTRY:
            raise ValueError(f"模型名重复: {name}")
        # 将模型类 `cls_or_fn` 存储到全局字典 `MODEL_REGISTRY` 中
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return deco# 返回内部函数


def build_model(config) -> nn.Module:
    """根据 config.model 构建模型"""
    name = getattr(config, "model", None)
    if not name:
        raise KeyError("config.model 未设置")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 '{name}'，可选：{list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)# 取出模型类本身将 config 对象作为参数传递给这个类的构造函数 __init__，将创建好的模型实例返回


class BaseNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本文件 (model.py) 所在的目录的绝对路径
    
    # 模型的前向传播
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_tensor: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError
        # BaseNERModel 自己并没有实现前向传播的具体逻辑。如果你试图直接创建一个 BaseNERModel 的实例并调用它的 forward 方法，程序会立即抛出一个 NotImplementedError 错误并停止。


# ====================== 对抗训练 FGM  ======================
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    # 加扰动
    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 寻找包含 embedding 的参数名（RoBERT 的 embedding 层通常叫 word_embeddings）
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 保存原始参数
                self.backup[name] = param.data.clone()# 克隆
                # 计算扰动：r = epsilon * g / ||g||²
                norm = torch.norm(param.grad)# 默认2范数
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at) # 加上扰动r
    # 恢复原始参数
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ---------- 门控机制 ----------
class GatedConcatFusion(nn.Module):
    """
    稳定融合 + token-level 相关性
    - 先 LN(text, img_ctx)，再 concat -> 线性 -> 门控
    - 返回: fused, rel  (rel∈[0,1], [B,T,1])
    """

    def __init__(self, hidden_dim: int, init_gate_bias: float = -1.5,
                 init_alpha: float = 0.02, rel_temp: float = 2.0):
        super().__init__()
        self.ln_t = nn.LayerNorm(hidden_dim)
        self.ln_v = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate.bias, init_gate_bias)
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.rel_temp = rel_temp

    def forward(self, text_feat, img_ctx):  # [B,T,H]
        # 分别对文本特征 t 和图像上下文特征 v 进行归一化。使它们的数值范围（均值和方差）变得一致，防止某一模态的特征在数值上主导融合过程，从而让训练更稳定
        t = self.ln_t(text_feat)
        v = self.ln_v(img_ctx)
        # 将归一化后的文本特征 t 和图像上下文特征 v 在最后一个维度（隐藏层维度）上进行拼接
        z = torch.cat([t, v], dim=-1)  # [B,T,2H]
        # rel_temp是一个温度系数。当 rel_temp > 1 时，它会使 sigmoid 的输出变得更平滑（即更接近0.5），让模型在训练初期不要对相关性做出过于“自信”的判断
        # 将归一化并拼接后的结果送入一个降维网络，然后除以一个温度系数
        # 如果某个 token 的 rel 值接近 1，说明图像信息对理解这个 token 很重要（比如 token 是 "猫"，图像里也有一只猫）
        # 如果 rel 值接近 0，说明图像信息与这个 token 无关
        rel = torch.sigmoid(self.rel_head(z) / self.rel_temp)  # [B,T,1]
        # 用刚刚计算出的相关性分数 rel 来逐元素乘以（加权）图像上下文特征 v
        # 对于相关性高的 token，rel 接近 1，v 的信息被完整保留。
        # 对于相关性低的 token，rel 接近 0，v 的信息几乎被“清零”。
        # 这样，模型就学会了只在需要的地方使用图像信息。
        v = v * rel
        # 拼接：原始文本特征 t 和经过相关性加权后的图像特征 v
        zf = torch.cat([t, v], dim=-1)
        # 初步融合特征：将拼接后的特征 zf（维度为 2*H）重新投影回原始的隐藏层维度 H
        fused = self.proj(zf)
        # 门控：将拼接后的特征 zf（维度为 2*H）重新投影回原始的隐藏层维度 H，使用 sigmoid 计算出一个门控信号 g，其形状为 [B,T,H]，每个值都在 (0, 1) 之间
        g = torch.sigmoid(self.gate(zf))
        # 返回：门控 g 作用于初步融合特征 fused，进行信息筛选，乘上一个0.02防止融合过程过于激进，加上原始的、未被归一化的文本特征 text_feat（残差连接）
        return text_feat + self.alpha * (g * fused), rel


class CrossAttentionBlock(nn.Module):
    """标准多头跨注意力（Q=text, K/V=image）+ FFN（前馈网络）
    跨：Query 来自文本，而 Key 和 Value 来自图像
        让文本序列中的每一个 token，能够去“审视”图像序列的信息，从而生成一个富含图像上下文的新文本表示
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # PyTorch 内置的多头注意力层
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, image_tokens, image_mask: Optional[torch.Tensor] = None, return_weights=False):
        # text_feat: 文本特征， [批量大小 B, 文本序列长度 T, 隐藏层维度 H]
        # image_tokens: 图像特征，[批量大小 B, 图像 token 数量 K, 隐藏层维度 H]。这是 VisualResampler 的输出，可以看作是 K 个代表图像关键区域的“视觉词向量”。
        # image_mask: (可选) 图像掩码，用于指示 image_tokens 中的哪些是 padding（填充）的，在注意力计算中应被忽略
        
        # need_weights=return_weights 是否返回注意力矩阵 [B, Heads, T_text, K_img]
        # attn_out: 注意力输出 [B, T, H]
        # attn_weights: 注意力权重矩阵 [B, T, K] (PyTorch 默认平均了多头)
        attn_out, attn_weights = self.attn(query=text_feat, key=image_tokens, value=image_tokens,
                                           key_padding_mask=image_mask,
                                           need_weights=return_weights)  # image_mask: True  图像中 padding（填充）的，在注意力计算中应被忽略
        
        # text_feat + attn_out: 残差连接————保证了模型不会在信息提取过程中丢失原始的文本信息。即使注意力层没学到任何有用的东西（attn_out 接近于0），原始信息也能无损地传递下去
        # self.norm1 = nn.LayerNorm(hidden_dim)层归一化
        x = self.norm1(text_feat + attn_out)
        # 前馈网络：对融合了图像信息的特征 x 进行一次非线性变换。模型增加了表达能力，使其能学习更复杂的关系。
        # x -> Linear(H -> 4*H) -> ReLU -> Linear(4*H -> H)
        f = self.ffn(x)
        # 残差连接+层归一化
        x = self.norm2(x + f)
        
        if return_weights:
            return x, attn_weights
        return x


class VisualResampler(nn.Module):
    """一个 Vision Transformer 模型可能会将一张图片分割成 14x14 = 196 个 patch，再加上一个 [CLS] token，总共输出 197 个特征向量。如果后续的 CrossAttentionBlock 需要让文本中的每个 token 都去关注这 197 个视觉特征，计算量会非常大 (文本长度 * 197)
    信息摘要：VisualResampler 把这 197 个特征向量“榨取”出最重要的信息，然后用一小组（比如 8 个）新的可学习向量来表示这些信息
    """

    def __init__(self, hidden_dim: int, num_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 创建了一个形状为 [num_queries, hidden_dim] (例如 [8, 768]) 的随机张量，然后利用Parameter包装成一个模型的可学习参数
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, image_feat, image_mask: Optional[torch.Tensor] = None):
        # image_feat来自Vision Transformer 模型的图像 patch 特征，形状 [批量大小 B, patch数量 R, 隐藏层维度 H]，例如 [16, 197, 768]
        B, _, H = image_feat.shape
        # self.queries 的原始形状是 [K, H] ([8, 768])。 unsqueeze(0) 在第 0 维增加一个维度，变成 [1, K, H]
        # .expand(B, -1, -1)表示   B: 将第 0 维从 1 扩展到 B (批量大小)  -1: 表示该维度保持不变
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B,K,H]
        # 每一个“探针” q ，和所有的图像 patch 特征 image_feat (Key) 计算注意力分数。
        # 这个分数决定了每个图像 patch 对于回答这个“探针”的问题有多重要。
        # 然后，用这些分数对所有的 image_feat (Value) 进行加权求和。得到一个结果向量，这个向量是所有图像 patch 信息根据该“探针”的特定“视角”聚合而成的“答案”
        # attn_output（我们想要的）: 是 attn_output_weights * V 的值 (并考虑了多头合并)。它是最终的特征输出
        out, _ = self.attn(query=q, key=image_feat, value=image_feat, key_padding_mask=image_mask)
        return self.ln(out)


def compute_alignment_loss(text_ctx, fused, mask=None):
    """逐 token 余弦相似度对齐（可用实值 mask 作为权重）"""
    # text_ctx: 纯文本编码器输出的特征，形状为 [Batch_size, Sequence_length, Hidden_dim]代表了每个 token 原始的文本语义。
    # fused: 文本特征与图像特征融合后的特征，形状也是 [B, T, H]。它代表了每个 token 融合了多模态信息后的语义。
    # mask: 可选的权重掩码，形状为 [B, T]。用来指定哪些 token 的对齐更重要（权重更大），或者忽略某些 token（权重为0，如 padding）
    
    # 为什么归一化：余弦相似度的计算只关心向量的方向，不关心它们的大小（模长）。通过归一化，我们可以简化计算，并确保损失函数只惩罚方向上的差异
    t = F.normalize(text_ctx, dim=-1)# 对 text_ctx 张量的最后一个维度（即 Hidden_dim 维度）进行 L2 归一化
    v = F.normalize(fused, dim=-1)
    # 对于两个单位向量 a 和 b，它们的点积 sum(a * b) 就等于它们之间夹角的余弦值 cos(θ)
    # cos 是一个形状为 [B, T] 的张量，其中 cos[i, j] 的值就是第 i 个样本中第 j 个 token 的原始文本向量和融合后向量之间的余弦相似度
    cos = (t * v).sum(-1)  # [B,T]
    # 将余弦相似度转换为一个损失值
    # 当 cos 为 1 时，loss 为 1.0 - 1.0 = 0。这是最小的损失，表示对齐得很好。
    # 当 cos 变小时（方向偏离），1.0 - cos 的值就会变大，损失也就变大。
    loss = 1.0 - cos
    # 如果提供了 mask，就用它来对损失进行加权
    if mask is not None:
        # 如果 mask 中某个位置是 1，该位置的 loss 不变。
        # 如果 mask 中某个位置是 0（例如 padding token），该位置的 loss 变为 0，在最终计算中被忽略。
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean() # 返回平均损失


# ====================== 辅助：对齐损失v2 ======================
def compute_alignment_loss_v2(text_ctx, fused, mask=None, beta: float = 0.3):
    """
    原始 compute_alignment_loss 的局限: 它只关心方向。这意味着，即使 fused 向量的长度变得非常大或非常小，只要它的方向和 text_ctx 一致，损失就会为零。在深度学习中，特征向量的幅度（norm）通常也携带着信息，幅度的剧烈变化可能导致训练不稳定。
    组合：cosine + beta * MSE，用于稳住融合方向并轻微惩罚幅度偏移
    beta: 一个浮点数，用于平衡余弦损失（方向）和 MSE 损失（幅度）的重要性
    text_ctx, fused: [B,T,H]
    mask: [B,T] 实值权重
    """
    t = F.normalize(text_ctx, dim=-1)
    v = F.normalize(fused, dim=-1)
    cos = (t * v).sum(-1)
    # 前面都一样，只有这里有变化
    # (1.0 - cos): 这是余弦相似度损失，和 v1 版本一样
    # F.mse_loss(fused, text_ctx, reduction='none')：计算均方误差，fused 和 text_ctx 是未经归一化的原始向量，reduction='none'告诉 PyTorch 不要对损失进行求和或求平均，而是返回一个与输入形状相同的张量
    # .mean(-1)  最后一个维度（H 维度）求平均，[B, T] 每个元素代表了一个 token 的平均均方误差
    loss = (1.0 - cos) + beta * F.mse_loss(fused, text_ctx, reduction='none').mean(-1)
    if mask is not None:
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss


# ====================== Span 头 ======================
class SpanHead(nn.Module):
    def __init__(self, hidden: int, num_types: int = 4, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.start_fc = nn.Linear(hidden, 1)
        self.end_fc   = nn.Linear(hidden, 1)
        # 简单用 [h_start; h_end] 做类型分类；也可以换成 span 内池化
        self.type_fc  = nn.Linear(hidden * 2, num_types)

        # 初始化更稳一点
        nn.init.normal_(self.start_fc.weight, std=0.02)
        nn.init.zeros_(self.start_fc.bias)
        nn.init.normal_(self.end_fc.weight, std=0.02)
        nn.init.zeros_(self.end_fc.bias)
        nn.init.normal_(self.type_fc.weight, std=0.02)
        nn.init.zeros_(self.type_fc.bias)

    def forward(self, enc, mask):
        """
        enc: [B,T,H]（你融合后的 token 表征）
        mask: [B,T]  (1=有效)
        return:
            start_logits, end_logits: [B,T]
        """
        # 通过 Dropout 层，进行正则化
        x = self.drop(enc)
        # 对最后一个维度（H）进行操作，输出的张量形状为 [B, T, 1]，然后.squeeze(-1)移除张量中最后一个维度，变成 [B, T]
        # 序列中每个 token 作为实体起始位置和结束位置的 logits
        start = self.start_fc(x).squeeze(-1)
        end   = self.end_fc(x).squeeze(-1)
        # 也可在这里对 pad 位置置极小值；训练里会用 mask
        return start, end

def info_nce(z1, z2, tau=0.15):
    """批内 InfoNCE，z1/z2: [B,H] 已归一化"""
    # z1（文本特征）, z2(图像特征): 形状都是 [B, H]。B 是批次大小，H 是特征维度。
    # 一个重要的前提是，这两个张量在输入前通常已经被 L2 归一化（变成了单位向量）
    # tau: 温度超参数
    # 低温度 (tau < 1): 会放大相似度分数之间的差异。相似度高的会变得更高，低的会变得更低。这使得模型更容易区分正负样本，但可能导致对负样本的“惩罚”过重，训练不稳定。
    # 高温度 (tau > 1): 会平滑相似度分数，使它们的差异变小。
    
    
    # 配对分类器
    # 矩阵乘法:z1乘z2的转置，除以温度超参数
    # 假设批次大小 B = 4   
    # z1 包含4个句子的向量：[s1, s2, s3, s4]
    # z2 包含4个图像的向量：[p1, p2, p3, p4]
    # logits 是一个形状为 [B, B] 的矩阵
    # 对角线元素 logits[i, i] 代表(s1·p1, s2·p2, s3·p3, s4·p4) 是正样本对的相似度
    # 角线元素 logits[i, j] (i != j) 代表 (s1·p2, s3·p4 等) 负样本对的相似度
    logits = (z1 @ z2.t()) / tau
    # 伪标签：它不是从数据集中标注来的，而是我们根据批次内数据的排列顺序构造出来的
    # z1.size(0) 批次大小 B
    # arange 创建一个从 0 到 B-1 的整数序列：[0, 1, 2, ..., B-1]代表第0个句子对应的正样本是第0个图像
    labels = torch.arange(z1.size(0), device=z1.device)
    # 交叉熵损失：接收 4x4 的 logits 矩阵和 1x4 的 labels 向量
    # 对于第0行 logits[0, :]，它会计算其与目标 labels[0] (即 0) 之间的交叉熵损失。这个损失会促使 logits[0, 0] 的值变大，而 logits[0, 1], logits[0, 2], logits[0, 3] 的值变小。
    # 对于第1行 logits[1, :]，它会计算其与目标 labels[1] (即 1) 之间的交叉熵损失。这会促使 logits[1, 1] 的值变大，其他值变小。
    return F.cross_entropy(logits, labels)


def _resolve_path(script_dir: str, path: str) -> str:
    local = os.path.join(script_dir, path)
    # 如果本地没有找到模型文件夹，它会返回原始的模型名称（例如 roberta-base）
    return local if os.path.exists(local) else path


# ---------- 主模型（不使用 label_names/BIO约束/边界辅助） ----------
@register_model("MNER")
class MultimodalNER(BaseNERModel):
    """
    ViT → Resampler(K) → Cross-Attn(Q=text,K/V=image) → 相关性门控融合 → (可选)BiLSTM(pack) → LN+Linear → CRF
    损失 = CRF(token平均) + λ_align * 对齐 + λ_preserve * 保真 + λ_nce * InfoNCE + λ_sparse * mean(rel)
    - 不依赖 label_names，不设置 BIO 硬约束/边界辅助
    """

    def __init__(self, config):
        super().__init__(config)
        self.text_encoder_path = config.text_encoder
        self.image_encoder_path = config.image_encoder
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        self.dropout_rate = config.drop_prob
        self.use_image = config.use_image
        self.use_bilstm = config.use_bilstm
        self.resampler_tokens = config.resampler_tokens
        self.cross_attn_heads = config.cross_attn_heads
        self.align_lambda = getattr(config, "align_lambda", 0.05)
        self.vision_trainable = getattr(config, "vision_trainable", False)

        # 训练/稳定化超参（均可从 config 覆盖）
        self.align_warmup_epochs = getattr(config, "align_warmup_epochs", 5)
        self.preserve_lambda = getattr(config, "preserve_lambda", 0.05)
        self.nce_lambda = getattr(config, "nce_lambda", 0.02)
        self.sparsity_lambda = getattr(config, "sparsity_lambda", 0.01)  # rel 稀疏正则
        self.image_dropout_p = getattr(config, "image_dropout_p", 0.3)  # 2015:0.3, 2017:0.2/0
        self.emission_temperature = getattr(config, "emission_temperature", 2.5)
        self.current_epoch = 0  # 由训练循环注入

        # ---- 文本编码器 ----
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        if self.text_encoder_path == "roberta-base":
            self.text_encoder = RobertaModel.from_pretrained(t_path)
        elif self.text_encoder_path == "bert-base-uncased":
            self.text_encoder = BertModel.from_pretrained(t_path)
        else:
            raise ValueError(f"Unsupported text encoder: {self.text_encoder_path}")
        self.text_hidden = self.text_encoder.config.hidden_size

        # ---- 视觉编码器（默认冻结）----
        v_path = _resolve_path(self.script_dir, self.image_encoder_path)# 如果模型文件夹不存在，返回名称字符
        # 触发自动下载
        self.clip = CLIPModel.from_pretrained(v_path)
        self.clip_vision = self.clip.vision_model
        # 不训练视觉模型
        if not self.vision_trainable:
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            self.clip_vision.eval()# 评估模式

        # ViT hidden -> text hidden
        self.clip_proj = nn.Linear(self.clip_vision.config.hidden_size, self.text_hidden)

        # ---- Resampler + Cross-Attn + 融合 ----
        self.dropout = nn.Dropout(self.dropout_rate)
        self.resampler = VisualResampler(self.text_hidden, num_queries=self.resampler_tokens,
                                         num_heads=self.cross_attn_heads, dropout=self.dropout_rate)
        self.cross_attn = CrossAttentionBlock(self.text_hidden, num_heads=self.cross_attn_heads,
                                              dropout=self.dropout_rate)
        self.fusion = GatedConcatFusion(self.text_hidden, init_gate_bias=-1.5, init_alpha=0.02, rel_temp=2.0)

        # ---- （可选）BiLSTM ----
        if self.use_bilstm:
            self.bilstm = nn.LSTM(input_size=self.text_hidden, hidden_size=self.hidden_dim // 2,
                                  num_layers=1, batch_first=True, bidirectional=True)
            out_dim = self.hidden_dim
        else:
            out_dim = self.text_hidden

        # ---- 分类层 + CRF（加LN，温和初始化）----
        self.classifier = nn.Sequential(
            # 将特征送入最后的线性层之前进行归一化，可以使训练过程更稳定，有时还能加速收敛
            # out_dim即前面网络（如 BiLSTM 或 Transformer）输出的每个 token 的向量表示维度
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.num_labels)
        )
        # 对classifier模块的Linear层手动初始化
        nn.init.normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

        # self.crf = CRF(self.num_labels, batch_first=True)
        # ===== Span 配置 =====
        self.use_span = getattr(config, "use_span", True)   # 打开后用 span 训练
        self.num_span_types = 4                             # LOC/ORG/OTHER/PER
        # 设置模型在预测时能够识别的实体的最大长度（token 数），有助于在解码阶段过滤掉不切实际的超长候选实体
        self.max_span_len = getattr(config, "max_span_len", 12)
        # lambda_type：类型分类损失的权重系数，Span 的总损失通常由三部分组成：loss_start + loss_end + lambda_type * loss_type
        self.lambda_type = getattr(config, "lambda_type", 1.0)
        # 可选：辅助交叉熵损失，即使主要任务是 Span 分类，我们仍然可以利用 self.classifier 计算一个传统的 Token 级别分类损失，并将其以一个较小的权重（aux_ce_lambda）加入到总损失中
        # 在训练早期，直接学习 Span 预测可能比较困难。这个辅助的、更简单的 Token 分类任务可以提供额外的监督信号，帮助模型的底层表示学得更好，从而稳定和加速整体训练过程。
        self.aux_ce_lambda = getattr(config, "aux_ce_lambda", 0.0)# 默认值为 0.0，表示不使用此辅助损失。

        # 实例化了 SpanHead 模块
        self.span_head = SpanHead(out_dim, num_types=self.num_span_types, dropout=self.dropout_rate)


    # 可选：仅解冻最后 n 个 ViT block
    def unfreeze_last_vision_blocks(self, n_blocks=2):
        # n_blocks要解冻的最后几个块（block）的数量
        total = len(self.clip_vision.encoder.layers)
        for i, blk in enumerate(self.clip_vision.encoder.layers):
            for p in blk.parameters():
                p.requires_grad = (i >= total - n_blocks)
        # 更新模型的状态标志 self.vision_trainable 为 True
        self.vision_trainable = True
        self.clip_vision.train(True)# 将视觉模型切换到训练模式

    def forward(self,
                input_ids,
                attention_mask,
                image_tensor=None,
                labels=None,# # [B, T] token级别的BIO标签ID，可选，用于辅助损失或旧评测
                # ===== 新增：span 监督 =====
                span_starts=None,  # [B,S_max]  真实实体span的起始位置
                span_ends=None,    # [B,S_max]  真实实体span的结束位置 (右开区间)
                span_types=None,   # [B,S_max]  0..3 (LOC/ORG/OTHER/PER)，无效填 -1
                span_mask=None,     # [B,S_max]  标记哪些span是有效的 (1/0)
                # ===== 修改：透传开关，允许在推理时返回权重 =====
                return_weights=False
                ):
        # 1) 文本编码
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]获取最后一层的隐藏状态，即每个token的特征向量
        txt = self.dropout(txt)
        # 初始化所有损失为0
        align_loss = torch.tensor(0.0, device=txt.device)# 对齐损失
        preserve   = torch.tensor(0.0, device=txt.device)# 保真损失
        nce_loss   = torch.tensor(0.0, device=txt.device)# 对比学习损失
        sparsity   = torch.tensor(0.0, device=txt.device)# 稀疏性损失
        
        # 初始化融合后的特征 (fused) 为纯文本特征 (txt)
        # 如果后续没有图像处理，fused 就等于 txt
        fused = txt
        
        # 预留注意力权重变量
        attn_weights_out = None

        # 2) 可选图像路径（带图像dropout）
        use_img = self.use_image and (image_tensor is not None)
        # 以一定的概率随机决定“丢弃”本次的图像输入。正则化手段，强迫模型不能过度依赖图像，增强鲁棒性。
        if self.training and use_img and (self.image_dropout_p > 0):
            # 一个从 [0, 1) 区间（即包含0，不包含1）的均匀分布中随机抽取的浮点
            if torch.rand(1, device=txt.device) < self.image_dropout_p:
                use_img = False

        if use_img:
            # with torch.set_grad_enabled(True):启动梯度计算
            with torch.set_grad_enabled(self.vision_trainable):
                
                
                # 将图像张量送入CLIP的视觉模型（通常是ViT）
                v_out = self.clip_vision(pixel_values=image_tensor)
                # ViT的输出中，第一个token是[CLS]全局特征，后面的是每个图像块(patch)的特征，[:, 1:, :]从第二个token开始取到最后，即丢掉[CLS] token
                patches = v_out.last_hidden_state[:, 1:, :]
            # 将从CLIP出来的视觉特征维度(H_vision)通过一个线性层(self.clip_proj)投影到和文本特征维度(H)一致
            img = self.clip_proj(patches)       # [B,R,H]
            # 对图像特征进行随机失活，防止过拟合
            img = self.dropout(img)
            # 信息摘要 (Resampling)
            img_tokens = self.resampler(img)    # [B,K,H]
            
            # 跨模态交互
            if getattr(self.config, "output_attn", False) or return_weights:
                txt_ctx, attn_map = self.cross_attn(txt, img_tokens, return_weights=True)
                attn_weights_out = attn_map
                # 如果需要返回权重，可以在这里拦截返回，或者存入 self 供外部调用
                # 这里为了不破坏原有逻辑，简单处理：如果只做推理可视化，我们可以专门写个 inference 方法
            else:
                txt_ctx = self.cross_attn(txt, img_tokens)
                
            # 门控融合：将原始文本特征`txt`和图像上下文`txt_ctx`送入`self.fusion`模块
            # fused[B,T,H]是最终融合后的特征，rel[B,T,1]是每个文本token与图像的相关性分数[0,1]
            fused, rel = self.fusion(txt, txt_ctx)
            # 稀疏性正则化：计算所有token相关性分数的平均值
            # 将这个值加入总损失，可以鼓励模型只在必要的时候才赋予较高的相关性(rel)，
            # 避免不相关的词也从图像中引入噪声。
            sparsity = rel.mean()                        # 标量

            # ===== 对齐（按相关性 + 实体权重；无 labels 时用 span 构造 entity_mask） =====
            # 如果是训练模式，并且对齐损失的权重系数 `align_lambda` > 0
            if self.training and (self.align_lambda > 0):
                # 相关性分数rel.detach()表示这部分不参与反向传播，只作为权重使用。
                rel_w = rel.detach().squeeze(-1) # [B,T]
                # entity_mask：优先用 labels，否则用 span 反投影
                if labels is not None:
                    # O标签ID为0，其他都>0
                    entity_mask = (labels > 0).float()
                elif (span_starts is not None) and (span_ends is not None) and (span_mask is not None):
                    # entity_mask 的形状和输入的文本序列一样，其中实体词所在的位置被标记为 1.0，非实体词的位置则为 0.0
                    entity_mask = torch.zeros_like(attention_mask, dtype=torch.float)# [B, T]，T 是序列长度
                    B, T = attention_mask.size()
                    
                    # 逐一处理批次中的每一条样本（句子）
                    for b in range(B):
                        # 生成一个布尔张量，`True` 的位置表示这是一个有效的 span
                        valid = span_mask[b] == 1
                        # 当前样本所有真实实体的起始位置,结束位置
                        ss = span_starts[b][valid]
                        ee = span_ends[b][valid]
                        # `zip` 将 `ss` 和 `ee` 配对，比如 `(start1, end1)`, `(start2, end2)`, ...
                        for s, e in zip(ss.tolist(), ee.tolist()):
                            # s是start1 e是end1
                            s = int(s); e = int(e)
                            if 0 <= s < e <= T:
                                entity_mask[b, s:e] = 1.0  # 将实体从头到尾设置为mask=1
                else:
                    entity_mask = torch.zeros_like(attention_mask, dtype=torch.float)
                
                # 对齐损失的权重： (1) 词与图的相关性，(2) 词是否为实体，以及 (3) 词是否为 padding
                # 将“相关性分数”和“实体掩码”相加：考虑了与图像的相关性rel_w，又给予了实体词一个额外的“优先处理权”
                    # 对于一个非实体词：它的 entity_mask 值为 0。所以它的权重就是它的相关性分数 rel_w
                    # 对于一个实体词：它的 entity_mask 值为 1.0。所以它的权重是 rel_w + 1.0。这意味着，实体词的权重被额外提升了 1.0。无论这个实体词与图像是否相关，它的基础权重至少是 1.0
                # 最后再乘attention_mask，所有 padding 位置的权重都被清零
                align_w = attention_mask.float() * (rel_w + entity_mask)
                # 计算两个特征向量 txt (原始文本特征) 和 fused (融合后特征) 之间的余弦相似度损失（方向）和MSE (均方误差)（幅度）
                # 传入align_w：权重高的 token (比如实体词) 对总损失的贡献更大，从而在反向传播时获得更大的梯度，促使模型优先优化它们。权重低的 token 的损失则会被相应减小。
                raw_align = compute_alignment_loss_v2(txt, fused, mask=align_w, beta=0.3)
                # 预热 (warm-up) 权重：在训练刚开始时，这个对齐损失的权重很小（接近0），让模型先集中精力学习更基本的任务。随着训练的进行，这个损失的权重逐渐增大，模型开始关注更精细的多模态对齐。
                # current_epoch当前训练周期，align_warmup_epochs总预热 epoch
                warm = min(1.0, getattr(self, "current_epoch", 0) / max(1, self.align_warmup_epochs))
                # 最终的、经过预热调整后的对齐损失
                align_loss = warm * raw_align

            
            # 计算 fused 特征和 txt 特征在每个 token 位置上的均方误差 (MSE)
            # .mean(-1)对 H 维度求平均，就得到了每个 token 的平均均方误差
            preserve_map = F.mse_loss(fused, txt, reduction='none').mean(-1)   # [B,T]
            
            # 保真损失
            # 教会模型在“应该”忽略图像的时候，学会“主动”忽略图像，从而保证融合过程是“按需服务”，而不是“盲目混合”
            # align_loss 的作用是拉近 fused 和 txt，鼓励它们在方向上保持一致
            # “相关性”分数,1.0 - rel.squeeze(-1)反转了权重，越不相关，权重越高
            # preserve_map: 基础的误差地图
            # attention_mask.float(): 过滤掉 padding 位置（将其权重设为0）
            # attention_mask.sum()  计算整个批次中有效 token 的总数
            preserve = (preserve_map * attention_mask.float() * (1.0 - rel.squeeze(-1))).sum() \
                       / (attention_mask.sum() + 1e-6)

            # 让模型学会对齐文本和图像的语义
            # 如果一个句子和一张图片是配对的（即它们描述的是同一件事），那么它们的全局特征向量应该互相靠近。
            # 如果它们是不配对的（即来自批次中不同的样本），那么它们的特征向量应该互相远离。
            # nce_lambda 是 InfoNCE 损失的权重系数
            if self.training and (self.nce_lambda > 0):
                # 每个样本的所有有效 token 的特征向量加在一起
                # 计算每个样本中有效 token 的数量
                # txt_pool形状是 [B, H]，每一行都是对应句子的全局语义表示
                txt_pool = (txt * attention_mask.unsqueeze(-1).float()).sum(1) \
                           / (attention_mask.sum(1, keepdim=True) + 1e-6)
                txt_pool = F.normalize(txt_pool, dim=-1)
                # 图像的全局语义表示：在 ViT 架构中，第 0 个 token 就是 [CLS] token，它被设计用来聚合整个图像的全局信息，类似于 BERT 中的 [CLS]
                v_global = v_out.last_hidden_state[:, 0, :]
                # 对图像表示也进行归一化，并在此之前通过 `clip_proj` 将其维度从 H_vision 对齐到 H
                v_global = F.normalize(self.clip_proj(v_global), dim=-1)
                # 调用 info_nce 辅助函数来计算损失
                nce_loss = info_nce(txt_pool, v_global, tau=0.15)

        # 如果配置中启用了BiLSTM层
        if self.use_bilstm:
            # 判断是否处于训练模式
            if self.training:
                # 在自然语言处理中，一个批次里的句子通常长度不同。为了把它们整理成一个矩形张量（比如 [batch_size, max_length, hidden_dim]），我们通常会用一个特殊的值（比如0）来填充（padding）那些较短的句子。
                # 然而，如果直接把这个带有很多 padding 的张量送入RNN（如LSTM）：
                # 结果污染：padding部分的计算结果可能会影响到后续时间步的隐藏状态，尤其是对于双向RNN的后向传播。
                # 1. 计算每个序列的真实长度
                lengths = attention_mask.sum(dim=1).cpu()
                # 2. pack_padded_sequence 打包：存储所有序列的有效部分，并记录了它们的原始结构信息
                packed = nn.utils.rnn.pack_padded_sequence(fused, lengths, batch_first=True, enforce_sorted=False)
                # 3. 送入BiLSTM，RNN内部就只会对真实的、非padding的数据进行计算
                # `packed_out` 同样是一个 PackedSequence 对象，包含了BiLSTM的输出
                packed_out, _ = self.bilstm(packed)
                # 4. pad_packed_sequence 解包
                # 保证张量形状能够对齐：`total_length=fused.size(1)` 指定了解压后的序列最大长度应该和输入时一样
                fused, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=fused.size(1))
            else:
                # 【导出友好模式】直接输入 padded tensor
                # LSTM 会处理所有的 token（包括 padding），结果虽然在 padding 处有值
                fused, _ = self.bilstm(fused)

        fused = self.dropout(fused)  # [B,T,H]
        B, T, H = fused.size()

        # ===== Span / Token 两种训练分支 =====
        # 初始化总损失=0
        total = torch.tensor(0.0, device=fused.device)

        # ---- 4.1 Span-based 主损失 ----
        # # 检查是否启用span模式，并且dataloader是否提供了span相关的标注信息
        if self.use_span and (span_starts is not None) and (span_ends is not None) and (span_types is not None) and (span_mask is not None):
            # 4.1.1 边界预测损失 (Start/End Prediction Loss)
            # 将融合后的特征 `fused` 送入 `span_head`，得到每个 token 作为 start 和 end 的 logits（未经sigmoid的原始输出）
            start_logits, end_logits = self.span_head(fused, attention_mask)  # [B,T]
            # 创建两个全零的目标矩阵，用于存放“真实”的 start 和 end 标签
            start_targets = torch.zeros_like(start_logits)# [B,T]
            end_targets   = torch.zeros_like(end_logits)# [B,T]

            # --- 构造真实的 start/end 目标矩阵 ---
            # `span_mask` 是一个0/1矩阵，标记了哪些提供的 span 标注是有效的（因为有padding）
            valid = span_mask == 1
            if valid.any():# 如果这个批次里至少有一个有效的真实 span
                # 高级索引
                # 沿着第二个维度进行“广播”或“复制”，使得新张量的形状与目标张量 span_starts 完全一样
                # b_idx = tensor([[0, 0, 0],
                #                 [1, 1, 1]])
                b_idx = torch.arange(B, device=fused.device).unsqueeze(1).expand_as(span_starts)
                # --- 构造 start 目标 ---
                ss = span_starts.clone() # 复制一份 start 位置
                ss[~valid] = -1    # 将无效 span 的位置设为-1，以便后续过滤
                mask_s = ss >= 0   # 创建一个布尔掩码，只保留有效的位置
                # 把start_targets[B,T]矩阵中对应实体起始位置的地方变成 1.0
                # b_idx 是 [[0, 0, 0], [1, 1, 1]]
                # mask_s 是 [[True, True, False], [True, False, False]]
                # b_idx[mask_s]取出的元素是：第0行第0个 (0)，第0行第1个 (0)，第1行第0个 (1)  即tensor([0, 0, 1])
                # ss[mask_s]表示取出ss中表示实体开头的索引
                start_targets[b_idx[mask_s], ss[mask_s]] = 1.0
                # end 使用 e-1（右开区间）
                # # end_targets[B,T]矩阵中对应实体结束位置的地方变成 1.0
                ee = (span_ends - 1).clamp(min=-1)
                ee[~valid] = -1
                mask_e = ee >= 0
                end_targets[b_idx[mask_e], ee[mask_e]] = 1.0

            # 只在有效 token 上计算 BCE
            weight_tok = attention_mask.float()# [B, T] (批次大小, 序列长度)
            # 只关心真实 token 的预测好坏，忽略模型在 padding 位置上的预测
            # start_logits：模型预测的每个 token 作为实体起始点的原始分数（logits）
            # start_targets：真实的标签矩阵[B, T]。在真实实体起始点的位置是 1.0，其他位置是 0.0
            # weight_tok：计算每个位置的 BCE 损失后，乘以对应的权重
            # reduction='sum'：不要自动计算平均损失，而是将所有位置（经过加权后）的损失全部加起来，得到一个总的损失和
            loss_start = F.binary_cross_entropy_with_logits(
                start_logits, start_targets, weight=weight_tok, reduction='sum'
            ) / (weight_tok.sum() + 1e-6)
            loss_end = F.binary_cross_entropy_with_logits(
                end_logits, end_targets, weight=weight_tok, reduction='sum'
            ) / (weight_tok.sum() + 1e-6)

            # 4.1.2 类型分类
            # 在这部分代码之前，模型已经通过 loss_start 和 loss_end 学习了如何“框出”所有可能的实体边
            # 现在，我们需要对这些被“框出”的实体进行分类。训练时，我们只对真实的、有标注的实体 (正样本) 进行分类监督。
            
            # valid形状为 [B, S_max] 的布尔矩阵，标记了哪些 span 标注是有效的
            # 只有当一个 span 既是有效的 (非 padding) 又有真实的类型标注时，它在 pos_mask 中对应的位置才是 True
            pos_mask = valid & (span_types >= 0)
            if pos_mask.any():# pos_mask 中是否至少有一个 True
                # 将所有正样本 span 的信息（批次索引b_lin、start位置、end位置、类型）从二维矩阵中提取出来，并拉平成一维的、长度为 N 的张量，其中 N 是这个批次中正样本的总数
                b_lin = torch.arange(B, device=fused.device).unsqueeze(1).expand_as(span_starts)
                b_pos = b_lin[pos_mask]
                s_pos = span_starts[pos_mask]
                e_pos = (span_ends[pos_mask] - 1).clamp(min=0, max=T-1)
                t_pos = span_types[pos_mask]  # [N]

                hs = fused[b_pos, s_pos, :] # [N,H]  并行地取出了所有 N 个正样本 span 的起始 token 的特征向量
                he = fused[b_pos, e_pos, :] # [N,H]
                # hs (形状 [N, H]): 每一行都是一个实体的起始词的特征向量[0.1, -0.5, 1.2, ...].......
                # he (形状 [N, H]): 每一行都是一个实体的结束词的特征向量[0.9, 0.2, -0.8, ...]
                h_span = torch.cat([hs, he], dim=-1) # [N,2H]
                # 拼接后，h_span (形状 [1, 2*H]) 会变成：[0.1, -0.5, 1.2, ..., 0.9, 0.2, -0.8, ...]
                # 新向量 h_span 同时包含了实体的开始和结束信息，可以被送入后续的分类器来判断该实体的类型
                logits_type = self.span_head.type_fc(h_span)  # [N,4]
                loss_type = F.cross_entropy(logits_type, t_pos)
            else:
                loss_type = torch.tensor(0.0, device=fused.device)
            # 总损失
            span_loss = loss_start + loss_end + self.lambda_type * loss_type
            # 如果 lambda_type = 0.5，意味着模型在优化时会更侧重于先把边界找对（因为 loss_start 和 loss_end 的权重相对更高），类型分类的优先级稍低。
            # 如果 lambda_type = 2.0，则意味着类型分类任务更重要
        else:
            span_loss = torch.tensor(0.0, device=fused.device)

        # ---- 4.2 Token-based 辅助损失 ----
        # 让模型同时学习两种不同形式的 NER 任务（Span-based 和 Token-based），有时可以相互促进，提高模型的泛化能力。
        #  在训练早期，直接学习复杂的 Span 任务可能比较困难。引入一个简单的 Token 分类任务可以提供一个更平滑的学习信号，帮助模型更快地进入状态。
        
        
        # self.classifier: 线性层，将每个 token 的最终特征向量 fused (形状 [B, T, H]) 映射到每个 BIO 标签的得分，这个张量我们称之为 emissions 或 logits
        emissions = self.classifier(fused) / self.emission_temperature  # [B,T,C]
        
        if (labels is not None) and (self.aux_ce_lambda > 0.0):
            # self.aux_ce_lambda辅助损失的权重系数
            
            # 交叉熵 (Cross-Entropy, CE) 损失
            ce = F.cross_entropy(
                emissions.view(-1, self.num_labels),# 将 emissions 张量从 [B, T, C] 重塑成 [B*T, C]  每一行是一个 token 的分类 logits
                labels.view(-1),# 将 labels 张量从 [B, T] 拉平成 [B*T]
                reduction='none'
            ).view(B, T)# 将一维的损失张量再重塑回 [B, T] 的形状，这样每个元素 ce[i, j] 就代表了第 i 个样本的第 j 个 token 的 CE 损失
            # 所有有效 token 的平均 CE 损失
            ce_loss = (ce * attention_mask.float()).sum() / (attention_mask.sum() + 1e-6)
        else:
            ce_loss = torch.tensor(0.0, device=fused.device)

        total = span_loss + self.aux_ce_lambda * ce_loss \
                + self.align_lambda * align_loss \
                + self.preserve_lambda * preserve \
                + self.nce_lambda * nce_loss \
                + self.sparsity_lambda * sparsity
                # 主任务损失+辅助损失
                # 对齐损失
                # 保真损失
                # 对比损失
                # 稀疏性损失

        # ===== 5) 训练/推理返回 =====
        # 修改：支持返回权重供可视化
        if return_weights:
            return total, attn_weights_out
            
        if (labels is not None) or (span_starts is not None):
            # 训练/验证阶段：返回总损失
            return total
        else:
            # 推理：默认返回 token 分类（兼容老评测）。
            # 你也可以调用 self.predict_spans(...) 得到 spans，再自行转 BIO。
            return emissions.argmax(dim=-1)  # [B,T]
            
    @torch.no_grad()
    def predict_spans(self, input_ids, attention_mask, image_tensor=None,
                      topk_s: int = 8, topk_e: int = 8):
        """
        返回：List[List[(s,e,type_id,score)]]（每条样本一组）
        说明：简单贪心，限制最大长度 self.max_span_len，避免过多重叠
        """
        self.eval()# 将模型切换到评估模式。这会关闭 Dropout 和 BatchNorm 等只在训练时使用的层。
        
        # 文本/图像编码与融合与 forward 相同，但不做损失，仅取 fused
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        fused = self.dropout(txt)
        if self.use_image and (image_tensor is not None):
            with torch.set_grad_enabled(False):
                v_out = self.clip_vision(pixel_values=image_tensor)
                patches = v_out.last_hidden_state[:, 1:, :]
            img = self.dropout(self.clip_proj(patches))
            img_tokens = self.resampler(img)
            txt_ctx = self.cross_attn(fused, img_tokens)
            fused, _ = self.fusion(fused, txt_ctx)
            fused = self.dropout(fused)

        start_logits, end_logits = self.span_head(fused, attention_mask)
        ps = torch.sigmoid(start_logits) * attention_mask
        pe = torch.sigmoid(end_logits)   * attention_mask

        B, T, H = fused.size()
        results = []
        # 遍历批次中的每个样本
        for b in range(B):
            # (attention_mask[b]==1).sum().item()计算当前样本的真实长度
            # .indices.tolist()获取这些位置的索引，并转换为 Python 列表
            # min(...)确保我们选取的 top-k 数量不会超过句子的实际长度
            s_idx = torch.topk(ps[b], k=min(topk_s, (attention_mask[b]==1).sum().item())).indices.tolist()
            e_idx = torch.topk(pe[b], k=min(topk_e, (attention_mask[b]==1).sum().item())).indices.tolist()
            
            cands = []# 初始化一个列表，存放所有合法的候选 span 及其分数
            # 双重循环，遍历所有 (start, end) 的组合
            for s in s_idx:
                for e1 in e_idx:
                    e = e1 + 1  # 右开区间
                    # --- 过滤不合法的 Span ---
                    # 长度过滤：结束点必须在起始点之后，且长度不能超过预设的最大值。
                    if (e - s) <= 0 or (e - s) > self.max_span_len:
                        continue
                    # Padding 过滤：确保整个 span [s, e) 范围内没有任何 padding。
                    # `attention_mask[b, s:e].min().item()` 如果中间有0(padding)，min就是0。
                    if attention_mask[b, s:e].min().item() == 0:
                        continue
                    # --- 为合法的 Span 评分 ---
                    # 提取这个候选 span 的 start 和 end 特征
                    # fused: 之前得到的融合了图文信息的特征矩阵，形状是 [B, T, H]
                    hs = fused[b, s, :]; he = fused[b, e-1, :]
                    # 将起始词特征 hs 和结束词特征 he 拼接起来,送入SpanHead 模块里的类型分类器
                    logits_type = self.span_head.type_fc(torch.cat([hs, he], dim=-1))  # [4]
                    prob_type = F.softmax(logits_type, dim=-1)
                    t = prob_type.argmax().item()# 找到概率 prob_type 中最大值所在的索引
                    # ps[b, s].item()：从我们之前计算的起始概率矩阵 ps 中，取出当前 span 起始位置 s 的概率
                    # pe[b, e-1].item()：从结束概率矩阵 pe 中，取出当前 span 结束位置 e-1 的概率
                    # prob_type[t].item()：从类型概率分布 prob_type 中，取出被预测为最可能的那个类型 t 的概率
                    score = (ps[b, s].item()) * (pe[b, e-1].item()) * (prob_type[t].item())
                    cands.append((s, e, t, score))
                    """[
                        (10, 12, 1, 0.85),  # "White House", score=0.85
                        (10, 11, 1, 0.40),  # "White", score=0.40
                        (3, 5, 0, 0.92),    # "New York", score=0.92
                        (4, 5, 2, 0.35)     # "York", score=0.35
                        ]"""
            
            
            # 简单贪心去重
            cands.sort(key=lambda x: x[3], reverse=True)# 降序：接收列表中的一个元素 x (即一个元组 (s, e, t, score)) 作为输入,取出元组中的第四个元素，也就是 score
            picked = []
            used = torch.zeros(T, dtype=torch.bool)# 创建一个长度为 T (句子总长度) 的一维布尔张量，并全部初始化为 False
            for s, e, t, sc in cands:
                if used[s:e].any():# 如果 .any() 返回 True，就意味着当前这个候选 span 与一个已经被选中的、分数更高的实体有重叠
                    continue
                picked.append((s, e, t, sc))
                # 会在这里将实体设为True，后续任何与它有重叠的、分数更低的实体在检查时都会被正确地过滤掉
                used[s:e] = True
            results.append(picked)
        return results



@register_model("roberta_crf")
class RobertaCRF(BaseNERModel):
    """纯文本基线：Roberta + Linear + CRF"""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.label_names = getattr(config, "label_names", None)
        self.dropout_rate = config.drop_prob
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = RobertaModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(H, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(txt))
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


# =========================
#  BERT（不带 CRF）
#  需要：config.text_encoder (如 "bert-base-chinese"), config.num_labels, config.drop_prob
#  训练返回 CE loss（忽略 -100），推理返回 argmax 预测
# =========================
@register_model("bert")
class BERTOnly(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels# 标签的总数量
        self.drop_prob = config.drop_prob # Dropout的比率，用于防止过拟合
        
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.drop_prob)
        
        self.classifier = nn.Linear(H, self.num_labels)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        # 获取发射分数
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        if labels is not None:# --- 训练模式 ---
            # 如果提供了真实的标签(labels)，说明正在进行训练
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )# 填充位置的标签设置为 -100
            return loss
        return logits.argmax(-1)  # --- 预测模式 ---


# =========================
#  BERT-CRF
#  需要：config.text_encoder, config.num_labels, config.drop_prob
# =========================
@register_model("bert_crf")
class BERTCRF(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels# 标签的具体名字列表
        self.drop_prob = config.drop_prob
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)
        # 定义一个CRF层。
        # 学习标签之间的转移规则（如 B-PER -> I-PER 的概率很高）
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        # 获取发射分数
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        # CRF层需要一个布尔类型的掩码来知道哪些位置是有效的，需要计算
        mask = attention_mask.bool()
        if labels is not None:# --- 训练模式 ---
            # 如果提供了真实的标签(labels)，说明正在进行训练
            # 调用CRF层的forward方法，计算给定logits和真实标签序列的对数似然损失。
            # 我们的目标是最大化这个似然，等价于最小化它的相反数，所以返回一个负值，作为我们要在训练中最小化的损失(loss)
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)# --- 预测模式 ---
        # 调用CRF层的decode方法：使用维特比算法在所有可能的标签路径中，结合logits（发射分数）和CRF内部学到的转移规则，找出一条最优（得分最高）的标签路径并返回


# =========================
#  BERT-BiLSTM-CRF
#  需要：config.text_encoder, config.hidden_dim, config.num_labels, config.drop_prob
# =========================
@register_model("bert_bilstm_crf")
class BERTBiLSTMCRF(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        self.drop_prob = config.drop_prob
        self.text_encoder_path = config.text_encoder
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size

        self.bilstm = nn.LSTM(H, self.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(self.drop_prob)
        # 定义一个全连接线性层（分类头）
        # 将BiLSTM的输出（维度是 self.hidden_dim）映射到每个标签的得分
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        # 定义一个CRF层：学习标签之间的转移规则，并帮助解码出最优的标签序列
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        # 取bert的last_hidden_state
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H_bert]
        # 送入 BiLSTM
        x, _ = self.bilstm(x)  # [B,T,H]
        # 送入分类头
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        # CRF层需要一个布尔类型的掩码来知道哪些位置是有效的，需要计算
        mask = attention_mask.bool()
        if labels is not None: # --- 这是训练模式 ---
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)