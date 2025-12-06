# -*- coding: utf-8 -*-
import torch
import os
import argparse
from model import build_model
from config import get_config
from test import load_config

script_dir = os.path.dirname(os.path.abspath(__file__))

def export_to_onnx(save_name, output_name="mnre_model.onnx"):
    """
    将模型导出为 ONNX 格式，用于加速推理部署
    """
    print(f"🚀 开始导出模型: {save_name} -> {output_name}")
    
    # 1. 加载配置和恢复模型
    # 注意：为了导出，我们通常使用 CPU 或者单卡
    device = torch.device("cpu")
    
    try:
        # 尝试加载训练时的配置
        config = load_config(save_name)
    except Exception as e:
        print(f"加载配置失败，使用默认配置: {e}")
        config = get_config()
    
    # 强制设置部分参数
    config.model = "MNER"
    config.device = "cpu"
    config.use_image = True
    # 模拟 num_labels (因为 build_model 需要)
    # 实际部署时应保证和训练一致
    if not hasattr(config, 'num_labels'):
        config.num_labels = 20 # 占位
        
    model = build_model(config).to(device)
    
    # 加载权重
    model_path = os.path.join(script_dir, "save_models", save_name, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print("权重加载成功！")
    else:
        print("⚠️ 未找到权重文件，将导出未训练的模型（仅用于测试结构）")

    model.eval()

    # 2. 构造 Dummy Input (虚拟输入)
    # 模拟 Batch=1, Seq=16 的输入
    dummy_input_ids = torch.randint(0, 1000, (1, 16), dtype=torch.long).to(device)
    dummy_mask = torch.ones((1, 16), dtype=torch.long).to(device)
    # 模拟图片 (1, 3, 224, 224)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)

    # 3. 执行导出
    output_path = os.path.join(script_dir, output_name)
    
    # 注意：forward 函数参数需要和这里对应
    # input_ids, attention_mask, image_tensor=None, ...
    
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_mask, dummy_image), # 传入参数元组，对应 forward 的前三个参数
            output_path,
            export_params=True,        # 存储权重
            opset_version=14,          # ONNX 版本，11-14 均可
            do_constant_folding=True,  # 优化常量折叠
            input_names=['input_ids', 'attention_mask', 'image'],
            # 输出名称根据 forward 的返回值确定。如果是 span 模式，forward 在 eval 时返回 logits
            output_names=['token_logits'], 
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'image': {0: 'batch_size'},
                'token_logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"✅ ONNX 模型已成功导出至: {output_path}")
        
        # 4. 简单校验
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型结构校验通过！")
        
    except Exception as e:
        print(f"导出失败: {e}")
        print("提示：请确保 model.forward 在 eval 模式下返回的是 Tensor 而不是 Loss")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="训练好的模型目录名")
    args = parser.parse_args()
    
    export_to_onnx(args.save_name)