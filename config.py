# -*- coding: utf-8 -*-
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    
    # ==================== 1. 基础运行与环境设置 ====================
    parser.add_argument("--device", type=str, default="cuda:0")# 使用第一块 GPU
    parser.add_argument("--epochs", type=int, default=100)# 一个 epoch 指的是模型完整地遍历一次所有训练数据。
    parser.add_argument("--batch_size", type=int, default=128)# 模型在一次前向/后向传播中处理的样本数量
    parser.add_argument("--max_len", type=int, default=128)# 文本序列的最大长度。超过此长度的文本会被截断，不足的会被填充
    
    # ==================== 2. 核心训练超参数 ====================
    parser.add_argument("--drop_prob", type=float, default=0.3)# Dropout正则化   防止模型过拟合
    # 差分学习率   clip_lr    fin_tuning_lr    downs_en_lr
    parser.add_argument("--clip_lr", type=float, default=1e-5)# 视觉编码器 (CLIP) 的学习率,通常设置得非常小，因为它已经很强大
    parser.add_argument("--fin_tuning_lr", type=float, default=5e-5)# 文本编码器 (RoBERTa) 的学习率
    parser.add_argument("--downs_en_lr", type=float, default=3e-4)# 下游任务层（如融合层、分类器）的学习率，设置得最大，因为这些层是从零开始学习的
    parser.add_argument("--weight_decay_rate", type=float, default=0.05)# 权重衰减（L2 正则化）系数,减轻过拟合
    parser.add_argument("--clip_grad", type=float, default=2.0)# 梯度裁剪的阈值。当梯度的范数超过这个值时，会被缩放到这个值，以防止梯度爆炸问题。
    parser.add_argument("--warmup_prop", type=float, default=0.1)# 学习率预热比例。在训练开始的 10% 步数内，学习率会从一个很小的值线性增加到设定的初始值，之后再正常衰减。有助于训练初期的稳定。
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)# 梯度累积步数:模型会累积 2 个批次的梯度后再进行一次参数更新，等效批次大小为 batch_size * 2 = 128 * 2 = 256
    # ==================== 3. 早停 (Early Stopping) 策略 ====================
    parser.add_argument("--min_epoch_num", type=int, default=5)# 只有当模型5个epoch在验证集上损失不在下降时，才会停止训练
    parser.add_argument("--patience", type=float, default=0.00001)# F1 分数的最小提升阈值。如果新的 F1 分数没有比历史最佳 F1 至少高出 0.00001，就会被认为没有提升。
    parser.add_argument("--patience_num", type=int, default=20)# 耐心值。如果连续 20 个 epoch，验证集上的 F1 分数都没有显著提升，训练将提前终止。
    # ==================== 4. 模型与数据集选择 ====================
    parser.add_argument("--text_encoder", type=str, default="roberta-base")# 预训练文本模型的名称或路径。
    parser.add_argument("--image_encoder", type=str, default="openai/clip-vit-base-patch32")# 图像编码器模型的名称或路径
    parser.add_argument("--model", type=str, default="MNER",
                        help="可选：roberta_crf | bert_bilstm_crf | roberta_clip_coattn | MNER")# 要使用的模型架构
    parser.add_argument('--use_image', action='store_true', help='是否使用图像模态')# 是否使用图像模态
    parser.add_argument('--use_bilstm', action='store_true', help='是否使用双向LSTM')# 是否使用双向LSTM
    # 原有 parser.add_argument(...) 后面追加：
    parser.add_argument("--continue_train_name", type=str, default="None",
                        help="保存于 save_models/ 下的目录名，用于继续训练（加载权重或完整状态）")# 断点续训。如果提供一个之前保存的模型目录名，会加载该模型的权重和优化器状态，从上次中断的地方继续训练。

    # ==================== 5. 实验跟踪与日志 ====================                    
    parser.add_argument("--ex_project", type=str, default="MNER")
    parser.add_argument("--ex_name", type=str, default="first_smoke_test")
    parser.add_argument("--ex_nums", type=str, default="A0")
    
    # ==================== WandB 设置 ====================
    parser.add_argument("--use_wandb", action='store_true', help="是否开启 WandB 记录")
    parser.add_argument("--wandb_project", type=str, default="MNER", help="WandB 项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB 用户/组织名称")
    
    # ==================== 6. MNER 模型专属结构参数 ====================
    parser.add_argument("--hidden_dim", type=int, default=768)# 适配下游模块（如 BiLSTM）的隐藏层维度
    parser.add_argument("--resampler_tokens", type=int, default=8)# 8 个可学习的“视觉Token”：
    # 视觉编码器(CLIP)会将图片分割成很多个小块，比如 14x14=196 个，通过注意力机制，每个视觉Token会汇集它最感兴趣的视觉信息，形成一个高度浓缩的摘要向量，也就是一个视觉Token，
    # 无论原始图片被分割成多少个小块，我们最终只提炼出 8 个最具代表性的视觉摘要（Token）来参与后续的跨模态融合
    parser.add_argument("--cross_attn_heads", type=int, default=8)# 跨注意力模块 (CrossAttentionBlock) 中的多头注意力头数
    parser.add_argument("--vision_trainable", action='store_true', help='')# 是否对视觉编码器 (CLIP) 的参数进行微调。微调会消耗大量显存，但可能带来性能提升。
    
    # ==================== 7. 多任务损失与正则化权重 ====================
    # 对齐、保真、NCE损失控制
    parser.add_argument("--align_lambda", type=float, default=0.2)# 对齐损失权重：促使融合后的特征与原始文本特征在语义上对齐
    parser.add_argument("--preserve_lambda", type=float, default=0.05,
                        help="保真损失的权重系数")# 保真损失权重：要求在与文本不相关的区域，融合特征应保留原始文本信息
    parser.add_argument("--nce_lambda", type=float, default=0.02,
                        help="InfoNCE 损失的权重系数")#  InfoNCE 损失权重：用于句级图文对比学习，拉近匹配的图文对，推远不匹配的
    parser.add_argument("--sparsity_lambda", type=float, default=0.01,
                        help="rel 稀疏正则项权重")# 稀疏性损失权重：鼓励模型只关注图像中与文本相关的少数关键区域
    parser.add_argument("--aux_ce_lambda", type=float, default=0.5,
                        help="辅助 Token 级交叉熵损失权重")# 辅助损失，帮助稳定训练
    parser.add_argument("--lambda_type", type=float, default=1.0,
                        help="Span 类型分类损失权重")# Span 类型分类权重
    
    # ==================== 7. 训练稳定化技巧 ====================
    parser.add_argument("--align_warmup_epochs", type=int, default=5,
                        help="对齐损失 warmup 的前几轮")# 对齐损失的预热轮数。在前 5 个 epoch，对齐损失的权重会从 0 慢慢增加到 --align_lambda，防止在训练初期对模型造成过大冲击。
    parser.add_argument("--emission_temperature", type=float, default=2.5,
                        help="logits 输出的温度参数")
    # 发射分数的温度系数。
    # 假设对于一个词，模型预测它属于 B-PER, I-PER, O 的 logits 分别是 [5.0, 1.0, 0.5]，经过 Softmax可能是 [0.98, 0.01, 0.01]。这是一个非常“自信”的分布，模型几乎百分之百地确定它是 B-PER
    # 在使用 Softmax 之前，我们把所有的 logits 除以一个温度值 T，概率分布变得更平滑，模型虽然仍然认为 B-PER 的可能性最大，但也给予了其他选项更多的“考虑空间”
    parser.add_argument("--image_dropout_p", type=float, default=0.5,
                        help="图像路径使用的 dropout 概率")# 图像模态的 Dropout 概率。在训练时，有 30% 的概率会随机丢弃图像输入，强制模型不过分依赖图像，增强模型的鲁棒性。
    parser.add_argument("--unfreeze_last_vision_blocks", type=int, default=2,
                        help="微调时解冻最后的视觉 encoder block 数")# 当 --vision_trainable 开启时，这个参数控制只解冻并微调视觉编码器最后的 2 个 Transformer Block，而不是整个模型。这是一种高效的微调策略，可以在节省资源和提升性能之间取得平衡。

    # ==================== 8. 阶段三：性能调优与鲁棒性增强 (新增) ====================
    # 对抗训练：增强鲁棒性，模拟攻击
    parser.add_argument("--use_fgm", action='store_true', help="是否开启 FGM 对抗训练")
    parser.add_argument("--fgm_epsilon", type=float, default=1.0, help="FGM 扰动系数")
    
    # 混合精度训练 (AMP)：加速训练，减少显存
    parser.add_argument("--use_amp", action='store_true', help="是否开启 FP16 混合精度训练")
    
    # 可视化与调试：推理时是否返回注意力权重
    parser.add_argument("--output_attn", action='store_true', help="模型推理时是否输出注意力权重")

    return parser.parse_args()