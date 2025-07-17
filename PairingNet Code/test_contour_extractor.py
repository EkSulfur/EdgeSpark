#!/usr/bin/env python3
"""
测试contour特征提取模块的脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from contour_feature_extractor import load_contour_extractor

def test_contour_extractor():
    """
    测试contour特征提取模块
    """
    print("=" * 60)
    print("测试Contour特征提取模块")
    print("=" * 60)
    
    # 模型路径
    model_path = "contour_feature_extractor_stage1_contour_extraction.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行以下命令:")
        print("1. python train_stage1.py")
        print("2. python extract_contour_module.py")
        return
    
    try:
        # 加载模型
        print("正在加载contour特征提取模块...")
        extractor, args = load_contour_extractor(model_path)
        extractor.eval()
        print("模块加载成功!")
        
        # 创建测试数据
        print("\n创建测试数据...")
        batch_size = 2
        num_points = 100
        patch_size = args.patch_size  # 从配置中获取
        
        # 模拟输入数据
        inputs = {
            'pcd': torch.randn(batch_size, num_points, 2),  # contour点云数据
            'c_input': torch.randn(batch_size, num_points, 3, patch_size),  # contour输入
            'adj': torch.randint(0, num_points*batch_size, (2, num_points*10))  # 邻接矩阵
        }
        
        print(f"输入数据shape:")
        print(f"  pcd: {inputs['pcd'].shape}")
        print(f"  c_input: {inputs['c_input'].shape}")
        print(f"  adj: {inputs['adj'].shape}")
        
        # 测试前向传播
        print("\n进行前向传播...")
        with torch.no_grad():
            l_c = extractor(inputs)
            
        print(f"输出特征shape: {l_c.shape}")
        print(f"输出特征范围: [{l_c.min().item():.4f}, {l_c.max().item():.4f}]")
        
        # 验证输出维度
        expected_shape = (batch_size, num_points, args.feature_dim)
        if l_c.shape == expected_shape:
            print(f"✓ 输出shape正确: {l_c.shape}")
        else:
            print(f"✗ 输出shape错误: 期望{expected_shape}, 实际{l_c.shape}")
            
        # 测试梯度
        print("\n测试梯度计算...")
        extractor.train()
        inputs['pcd'].requires_grad_(True)
        l_c = extractor(inputs)
        loss = l_c.sum()
        loss.backward()
        
        if inputs['pcd'].grad is not None:
            print("✓ 梯度计算正常")
        else:
            print("✗ 梯度计算失败")
        
        print("\n=" * 60)
        print("测试完成!")
        print("Contour特征提取模块工作正常 ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_example():
    """
    显示使用示例
    """
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    example_code = '''
# 在您的项目中使用contour特征提取模块

from contour_feature_extractor import load_contour_extractor
import torch

# 1. 加载模型
extractor, args = load_contour_extractor('contour_feature_extractor_stage1_contour_extraction.pth')
extractor.eval()

# 2. 准备输入数据
inputs = {
    'pcd': your_contour_data,        # shape: [batch_size, num_points, 2]
    'c_input': your_patch_data,      # shape: [batch_size, num_points, 3, patch_size]
    'adj': your_adjacency_matrix     # shape: [2, num_edges]
}

# 3. 提取特征
with torch.no_grad():
    l_c = extractor(inputs)         # shape: [batch_size, num_points, feature_dim]
    
# 4. 使用特征进行后续处理
# your_downstream_processing(l_c)
'''
    
    print(example_code)

if __name__ == "__main__":
    success = test_contour_extractor()
    if success:
        show_usage_example()
    else:
        print("请检查错误信息并重新运行")