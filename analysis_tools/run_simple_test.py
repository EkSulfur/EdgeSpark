"""
简化版EdgeSpark快速测试脚本
用于验证网络架构和训练流程
"""
import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network_simple import SimpleEdgeSparkNet
from dataset_simple import create_simple_dataloaders
from train_simple import SimpleTrainer, create_simple_config

def test_network():
    """测试网络架构"""
    print("🧪 测试简化版网络架构...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建网络
    model = SimpleEdgeSparkNet(
        segment_length=64,
        num_segments=8,
        feature_dim=128,
        hidden_dim=128
    ).to(device)
    
    # 测试数据
    batch_size = 8
    points1 = torch.randn(batch_size, 800, 2).to(device)
    points2 = torch.randn(batch_size, 900, 2).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(points1, points2)
    
    print(f"✅ 网络输出形状: {output.shape}")
    print(f"✅ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return True

def test_dataset():
    """测试数据集"""
    print("🧪 测试数据集...")
    
    try:
        from dataset_simple import SimpleEdgeSparkDataset
        
        # 测试数据集
        dataset = SimpleEdgeSparkDataset(
            "dataset/train_set.pkl",
            max_points=1000,
            augment=True,
            negative_ratio=1.0,
            hard_negative_ratio=0.3
        )
        
        sample = dataset[0]
        print(f"✅ 数据集大小: {len(dataset)}")
        print(f"✅ 样本形状: {sample['source_points'].shape}, {sample['target_points'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        return False

def test_dataloader():
    """测试数据加载器"""
    print("🧪 测试数据加载器...")
    
    try:
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            "dataset/train_set.pkl",
            "dataset/valid_set.pkl",
            "dataset/test_set.pkl",
            batch_size=16,
            max_points=1000,
            num_workers=0  # 测试时不使用多进程
        )
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"✅ 训练批次: {len(train_loader)}")
        print(f"✅ 批次形状: {batch['source_points'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    print("🧪 测试训练步骤...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = SimpleEdgeSparkNet(
            segment_length=64,
            num_segments=8,
            feature_dim=128,
            hidden_dim=128
        ).to(device)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # 创建测试数据
        batch_size = 8
        points1 = torch.randn(batch_size, 800, 2).to(device)
        points2 = torch.randn(batch_size, 900, 2).to(device)
        labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        output = model(points1, points2)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        print(f"✅ 训练步骤完成, 损失: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 EdgeSpark简化版测试")
    print("=" * 50)
    
    tests = [
        ("网络架构", test_network),
        ("数据集", test_dataset),
        ("数据加载器", test_dataloader),
        ("训练步骤", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: 失败 - {e}")
        print()
    
    # 总结
    print("=" * 50)
    print("🎯 测试结果:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始训练了")
        print("💡 运行训练: uv run python train_simple.py")
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)