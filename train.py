# -*- coding: utf-8 -*-

import json
import os
import random
from datetime import datetime

import numpy as np
import wandb
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 引入混合精度训练工具
from torch.cuda.amp import autocast, GradScaler

from dataloader import NERDataset, collate_fn_span, DataProcessor
# 引入 FGM 类
from model import build_model, FGM
from test import evaluate_model
from test import load_config

script_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=42):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # numpy 随机种子
    torch.manual_seed(seed)  # CPU torch 随机种子
    torch.cuda.manual_seed(seed)  # GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    # 保证 CUDA 可复现（但可能会略微降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

# 保存模型
def save_model_checkpoint(model, optimizer, scheduler, config, save_dir, epoch, best_metric):
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    # 实现断点恢复的关键：
    # 保存模型权重、优化器状态和学习率调度器状态
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    
    # 保存此次训练的配置文件：模型结构和训练设定的所有超参数，比如text_encoder: "roberta-base"，hidden_dim: 768
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    # 保存此次训练的训练状态：记录了训练进行到了哪一步，以及目前最好的F1是多少，知道应该从第 epoch + 1 轮开始
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump({"epoch": epoch, "best_f1": best_metric}, f, indent=2)


# 加载模型
def load_model_checkpoint(model, optimizer, scheduler, load_dir):
    model.load_state_dict(torch.load(os.path.join(load_dir, "model.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(load_dir, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(load_dir, "scheduler.pt")))

    with open(os.path.join(load_dir, "training_state.json")) as f:
        state = json.load(f)

    return state["epoch"], state["best_f1"]


def train(config):
    print("train config:", config)
    # 根据时间和配置生成一个唯一的实验名称
    run_name = f"{datetime.now().strftime('%Y-%m-%d')}_train_{str(config.model)}_{config.ex_name}"
    # 设置模型保存目录
    save_dir = os.path.join(script_dir, "save_models", f"{run_name}")
    # 设置计算设备 (CPU 或 GPU)
    device = torch.device(config.device)
    
    data_dir = os.path.join(script_dir, 'data') # 主数据目录
    img_path = os.path.join(data_dir, 'ner_img') # 统一的图片目录
    
    
    # 初始化 WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=vars(config)
        )
    
    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),# 裁剪到CLIP模型需要的224x224
        transforms.ToTensor(),# 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # 实例化数据处理器                        
    processor = DataProcessor(data_dir, config.text_encoder)
    # 创建训练集 Dataset 和 DataLoader
    train_dataset = NERDataset(
        processor, transform, img_path=img_path, max_seq=config.max_len,
        sample_ratio=1.0, mode='train', return_span=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=collate_fn_span
    )
    # 创建验证集 Dataset 和 DataLoader
    val_dataset = NERDataset(
        processor,
        transform,
        img_path=img_path,
        max_seq=config.max_len,
        sample_ratio=1.0,
        mode='valid'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=4,
        pin_memory=True
    )
    
    # 关键步骤：在构建模型前，先计算标签数量并写入 config
    config.num_labels = len(train_dataset.label_mapping)
    print(f"Detected {config.num_labels} labels: {train_dataset.label_mapping}")
    
    # 如果continue_train_name参数没填，则认为是首次训练
    if config.continue_train_name == "None":
        model = build_model(config).to(device)
    # 断点续训逻辑
    else:
        SAVE_ROOT = os.path.join(script_dir, "save_models") 
        ckpt_dir = os.path.join(SAVE_ROOT, config.continue_train_name)
        prev_cfg = load_config(config.continue_train_name)
        model = build_model(prev_cfg).to(device)
    
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # ---- 差分学习率 ----
    param_roberta, param_clip, param_downstream = [], [], []
    for n, p in model.named_parameters():
        if n.startswith("text_encoder."):# 属于 RoBERTa
            param_roberta.append((n, p))
        elif n.startswith("clip_vision.") or n.startswith("clip.vision_model."):# 属于 CLIP
            param_clip.append((n, p))
        else:# 属于下游模块
            param_downstream.append((n, p))

    def build_groups(named_params, lr, wd):
        params_decay = [p for n, p in named_params
                        if p.requires_grad and not any(nd in n for nd in no_decay)]
        params_nodec = [p for n, p in named_params
                        if p.requires_grad and any(nd in n for nd in no_decay)]
        groups = []
        if len(params_decay) > 0:
            groups.append({"params": params_decay, "weight_decay": wd, "lr": lr})
        if len(params_nodec) > 0:
            groups.append({"params": params_nodec, "weight_decay": 0.0, "lr": lr})
        return groups

    optimizer_grouped_parameters = []
    optimizer_grouped_parameters += build_groups(param_roberta, config.fin_tuning_lr, config.weight_decay_rate)
    optimizer_grouped_parameters += build_groups(param_downstream, config.downs_en_lr, config.weight_decay_rate)
    if getattr(config, "vision_trainable", False):
        optimizer_grouped_parameters += build_groups(param_clip, config.clip_lr, config.weight_decay_rate)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    t_total = len(train_loader) // config.gradient_accumulation_steps * config.epochs
    warmup_steps = int(t_total * config.warmup_prop)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    
    start_epoch = 0 
    best_f1 = 0.0 
    patience_counter = 0 

    # 断点续训加载
    if config.continue_train_name != "None":
        load_dir = ckpt_dir
        model.load_state_dict(torch.load(os.path.join(load_dir, "model.pt")))
        optimizer.load_state_dict(torch.load(os.path.join(load_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(load_dir, "scheduler.pt")))
        with open(os.path.join(load_dir, "training_state.json")) as f:
            state = json.load(f)
        start_epoch = state["epoch"] + 1  # 从下一轮开始
        best_f1 = state["best_f1"]
    
    # ===== [New] 初始化混合精度 Scaler =====
    scaler = GradScaler(enabled=config.use_amp)
    
    # ===== [New] 初始化对抗训练工具 =====
    fgm = None
    if config.use_fgm:
        print(f"🦁 FGM Adversarial Training Enabled (epsilon={config.fgm_epsilon})")
        fgm = FGM(model)
    
    # 外层循环:控制训练的总轮次
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        # 更新当前epoch，供warmup使用
        model.current_epoch = epoch 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)
        
        for step, batch in enumerate(loop):
            # 获取数据
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("image", None)
            if images is not None: images = images.to(device)
            
            span_starts = batch["span_starts"].to(device)
            span_ends = batch["span_ends"].to(device)
            span_types = batch["span_types"].to(device)
            span_mask = batch["span_mask"].to(device)

            # ===== 混合精度前向传播=====
            with autocast(enabled=config.use_amp):
                loss = model(input_ids, attention_mask, image_tensor=images,
                             labels=labels,
                             span_starts=span_starts, span_ends=span_ends,
                             span_types=span_types, span_mask=span_mask)
                loss = loss / config.gradient_accumulation_steps
            
            # ===== 反向传播=====
            scaler.scale(loss).backward()

            # ===== 3. 对抗训练  =====
            if config.use_fgm:
                # 在 embedding 上加扰动
                fgm.attack(epsilon=config.fgm_epsilon, emb_name='word_embeddings')
                
                with autocast(enabled=config.use_amp):
                    # 再次前向计算 (带扰动的参数)
                    loss_adv = model(input_ids, attention_mask, image_tensor=images,
                                     labels=None,
                                     span_starts=span_starts, span_ends=span_ends,
                                     span_types=span_types, span_mask=span_mask)
                    loss_adv = loss_adv / config.gradient_accumulation_steps
                
                # 反向传播梯度 (累加到原有梯度上)
                scaler.scale(loss_adv).backward()
                
                # 恢复原始参数:用扰动的损失更新原始参数
                fgm.restore(emb_name='word_embeddings')

            # ===== 参数更新 =====
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Unscale 梯度以便进行裁剪
                scaler.unscale_(optimizer)
                
                # 梯度裁剪
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                
                # Step 更新 (通过 scaler)
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                optimizer.zero_grad()
                
                if config.use_wandb:
                    wandb.log({
                        "train/grad_norm": norm,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/step": epoch * len(train_loader) + step
                    })

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])

        # End of Epoch
        avg_loss = total_loss / len(train_loader)
        if config.use_wandb:
            wandb.log({"train/loss": avg_loss, "epoch": epoch})
        print(f"\nEpoch {epoch} Train Loss: {avg_loss:.4f}")

        # 在验证集上评估
        acc, f1, p, r = evaluate_model(model, val_loader, device, train_dataset.label_mapping)
        print(f"🎯Epoch {epoch} Eval F1: {f1:.4f} precision: {p:.4f} recall: {r:.4f} acc:{acc:4f}")
        
        if config.use_wandb:
            wandb.log({
                "eval/f1": f1,
                "eval/precision": p,
                "eval/recall": r,
                "eval/acc": acc,
                "epoch": epoch
            })

        # 早停与模型保存
        if f1 > best_f1 + config.patience:
            best_f1 = f1
            patience_counter = 0
            save_model_checkpoint(model, optimizer, scheduler, config, save_dir, epoch, best_f1)
            print(f"Model saved to {save_dir}")
        else:
            patience_counter += 1
            print(f"No improvement, patience {patience_counter}/{config.patience_num}")
            if epoch >= config.min_epoch_num and patience_counter >= config.patience_num:
                print("Early stopping triggered.")
                break
                
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    from config import get_config

    set_seed(42)
    config = get_config()
    train(config)