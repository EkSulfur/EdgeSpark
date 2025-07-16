#!/bin/bash

# EdgeSpark 快速测试脚本
# 使用较小的配置进行快速测试，确保代码正常运行

echo "EdgeSpark 快速测试脚本"
echo "====================="

# 1. 更新环境
echo "1. 更新Python环境..."
uv sync

if [ $? -ne 0 ]; then
    echo "环境更新失败，请检查依赖配置"
    exit 1
fi

# 2. 测试网络架构
echo ""
echo "2. 测试网络架构..."
uv run python network_improved.py

if [ $? -ne 0 ]; then
    echo "网络测试失败"
    exit 1
fi

# 3. 测试数据加载
echo ""
echo "3. 测试数据加载..."
uv run python dataset_loader.py

if [ $? -ne 0 ]; then
    echo "数据加载测试失败"
    exit 1
fi

# 4. 创建快速训练配置
echo ""
echo "4. 创建快速测试训练配置..."
cat > quick_train.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

from network_improved import EdgeSparkNet
from dataset_loader import create_dataloaders

def quick_train():
    """快速训练测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建小规模配置
    model = EdgeSparkNet(
        segment_length=16,  # 更小的段长度
        n1=8,              # 更少的采样数
        n2=8,
        feature_dim=128,   # 更小的特征维度
        hidden_channels=32,
        num_samples=2      # 更少的采样次数
    ).to(device)
    
    # 创建数据加载器（小批量）
    train_loader, val_loader, _ = create_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=4,      # 小批量
        max_points=500,    # 更少的点数
        num_workers=0      # 不使用多进程
    )
    
    # 简单训练循环
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    
    print("开始快速训练测试（3个批次）...")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只训练3个批次
            break
            
        source_points = batch['source_points'].to(device)
        target_points = batch['target_points'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        predictions = model(source_points, target_points)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  批次 {i+1}: 损失 = {loss.item():.4f}")
    
    print("快速训练测试完成！")
    
    # 测试验证
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        source_points = batch['source_points'].to(device)
        target_points = batch['target_points'].to(device)
        labels = batch['label'].to(device)
        
        predictions = model(source_points, target_points)
        val_loss = criterion(predictions, labels)
        
        print(f"验证损失: {val_loss.item():.4f}")
        print(f"预测概率: {predictions.squeeze().cpu().numpy()}")
        print(f"真实标签: {labels.squeeze().cpu().numpy()}")

if __name__ == "__main__":
    quick_train()
EOF

# 5. 运行快速训练测试
echo ""
echo "5. 运行快速训练测试..."
uv run python quick_train.py

if [ $? -ne 0 ]; then
    echo "快速训练测试失败"
    exit 1
fi

# 6. 清理临时文件
rm -f quick_train.py

echo ""
echo "============================================"
echo "快速测试完成！所有组件正常工作"
echo "============================================"
echo ""
echo "现在可以运行完整训练："
echo "  bash run_training.sh"