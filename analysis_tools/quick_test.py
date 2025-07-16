"""
快速测试脚本
"""
import torch
import sys
import os

def quick_test():
    """快速测试基本功能"""
    
    # 1. 测试PyTorch
    print("1. PyTorch测试...")
    print(f"   版本: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    
    # 2. 测试数据集文件
    print("\n2. 数据集文件检查...")
    datasets = ["dataset/train_set.pkl", "dataset/valid_set.pkl", "dataset/test_set.pkl"]
    for ds in datasets:
        exists = os.path.exists(ds)
        size = os.path.getsize(ds) / 1024 / 1024 if exists else 0
        print(f"   {ds}: {'✅' if exists else '❌'} ({size:.1f}MB)")
    
    # 3. 测试网络创建
    print("\n3. 网络创建测试...")
    try:
        from network_simple import SimpleEdgeSparkNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleEdgeSparkNet().to(device)
        
        # 测试前向传播
        batch_size = 2
        points1 = torch.randn(batch_size, 500, 2).to(device)
        points2 = torch.randn(batch_size, 600, 2).to(device)
        
        with torch.no_grad():
            output = model(points1, points2)
        
        print(f"   ✅ 网络创建成功")
        print(f"   ✅ 前向传播成功: {output.shape}")
        print(f"   ✅ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"   ❌ 网络测试失败: {e}")
        return False
    
    # 4. 测试数据加载
    print("\n4. 数据加载测试...")
    try:
        import pickle
        
        with open("dataset/train_set.pkl", 'rb') as f:
            data = pickle.load(f)
        
        print(f"   ✅ 数据加载成功")
        print(f"   ✅ 边缘点云数量: {len(data['full_pcd_all'])}")
        print(f"   ✅ 匹配对数量: {len(data['GT_pairs'])}")
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False
    
    print("\n🎉 快速测试完成！")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("✅ 可以开始训练了!")
        print("🚀 运行命令: uv run python train_simple.py --epochs 10")
    else:
        print("❌ 存在问题，请检查")