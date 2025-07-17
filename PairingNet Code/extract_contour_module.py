#!/usr/bin/env python3
"""
提取contour特征模块的完整脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import __init__
import torch
import math
from glob import glob
from utils import pipeline, config
from contour_feature_extractor import extract_contour_module, load_contour_extractor

def get_best_checkpoint(checkpoint_dir):
    """
    获取最佳检查点文件
    """
    checkpoints = glob(os.path.join(checkpoint_dir, "checkpoint_*.tar"))
    if not checkpoints:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到检查点文件")
    
    # 按epoch数排序，取最新的
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]

def main():
    # 获取配置参数
    opt = config.args
    
    # 设置实验名称
    exp_name = 'stage1_contour_extraction'
    
    # 检查点目录
    checkpoint_dir = f"./EXP/{exp_name}/checkpoint/"
    
    print("=" * 60)
    print("PairingNet Contour特征提取模块提取器")
    print("=" * 60)
    
    # 检查是否存在训练好的模型
    if not os.path.exists(checkpoint_dir):
        print(f"错误: 检查点目录不存在: {checkpoint_dir}")
        print("请先运行 train_stage1.py 训练模型")
        return
    
    try:
        # 获取最佳检查点
        best_checkpoint = get_best_checkpoint(checkpoint_dir)
        print(f"找到最佳检查点: {best_checkpoint}")
        
        # 设置保存路径
        save_path = f"./contour_feature_extractor_{exp_name}.pth"
        
        # 提取contour特征模块
        print("正在提取contour特征模块...")
        contour_extractor = extract_contour_module(best_checkpoint, opt, save_path)
        
        print("=" * 60)
        print("提取完成!")
        print(f"独立的contour特征提取模块已保存到: {save_path}")
        print("=" * 60)
        
        # 验证提取的模块
        print("正在验证提取的模块...")
        loaded_extractor, loaded_args = load_contour_extractor(save_path)
        print("模块加载成功!")
        
        # 显示使用示例
        print("\n使用示例:")
        print("```python")
        print("from contour_feature_extractor import load_contour_extractor")
        print("import torch")
        print()
        print("# 加载模型")
        print(f"extractor, args = load_contour_extractor('{save_path}')")
        print("extractor.eval()")
        print()
        print("# 准备输入数据")
        print("inputs = {")
        print("    'pcd': torch.randn(1, 100, 2),  # contour点云数据")
        print("    'c_input': torch.randn(1, 100, 3, 7),  # contour输入")
        print("    'adj': torch.randn(2, 1000)  # 邻接矩阵")
        print("}")
        print()
        print("# 提取特征")
        print("with torch.no_grad():")
        print("    l_c = extractor(inputs)  # 输出contour特征")
        print("    print(f'Contour特征shape: {l_c.shape}')")
        print("```")
        
    except Exception as e:
        print(f"错误: {e}")
        return

if __name__ == "__main__":
    main()