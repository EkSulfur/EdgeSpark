#!/usr/bin/env python3
"""
测试最终改进方案的基本功能
"""
import torch
import torch.nn as nn
import sys
import traceback

# 添加路径
sys.path.append('/home/eksulfur/EdgeSpark')

def test_network_basic_functionality():
    """测试网络基本功能"""
    print("🧪 测试最终改进方案基本功能")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 测试数据
    batch_size = 2
    points1 = torch.randn(batch_size, 100, 2).to(device)  # 较小的测试数据
    points2 = torch.randn(batch_size, 120, 2).to(device)
    
    results = []
    
    # 1. 测试高采样方案
    print("\n1️⃣ 测试高采样方案...")
    try:
        from final_improvements.high_sampling_approach import HighSamplingEdgeMatchingNet
        
        model = HighSamplingEdgeMatchingNet(
            segment_length=30,  # 较小的段落长度
            num_samples=5,      # 较少的采样数
            feature_dim=64      # 较小的特征维度
        ).to(device)
        
        with torch.no_grad():
            output = model(points1, points2)
        
        print(f"   ✅ 高采样方案测试成功")
        print(f"   输出形状: {output.shape}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        results.append(("高采样方案", True, sum(p.numel() for p in model.parameters())))
        
    except Exception as e:
        print(f"   ❌ 高采样方案测试失败: {e}")
        traceback.print_exc()
        results.append(("高采样方案", False, 0))
    
    # 2. 测试基础傅里叶方案
    print("\n2️⃣ 测试基础傅里叶方案...")
    try:
        from final_improvements.fourier_approach import FourierBasedMatchingNet
        
        model = FourierBasedMatchingNet(
            max_points=1000,
            num_freqs=32,       # 较少的频率分量
            feature_dim=64      # 较小的特征维度
        ).to(device)
        
        with torch.no_grad():
            output = model(points1, points2)
        
        print(f"   ✅ 基础傅里叶方案测试成功")
        print(f"   输出形状: {output.shape}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        results.append(("基础傅里叶方案", True, sum(p.numel() for p in model.parameters())))
        
    except Exception as e:
        print(f"   ❌ 基础傅里叶方案测试失败: {e}")
        traceback.print_exc()
        results.append(("基础傅里叶方案", False, 0))
    
    # 3. 测试混合傅里叶方案
    print("\n3️⃣ 测试混合傅里叶方案...")
    try:
        from final_improvements.fourier_approach import HybridFourierNet
        
        model = HybridFourierNet(
            max_points=1000,
            num_freqs=32,
            feature_dim=64,
            num_samples=3       # 较少的采样数
        ).to(device)
        
        with torch.no_grad():
            output = model(points1, points2)
        
        print(f"   ✅ 混合傅里叶方案测试成功")
        print(f"   输出形状: {output.shape}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        results.append(("混合傅里叶方案", True, sum(p.numel() for p in model.parameters())))
        
    except Exception as e:
        print(f"   ❌ 混合傅里叶方案测试失败: {e}")
        traceback.print_exc()
        results.append(("混合傅里叶方案", False, 0))
    
    # 总结测试结果
    print(f"\n📊 测试结果总结")
    print("=" * 60)
    
    successful_tests = [r for r in results if r[1]]
    failed_tests = [r for r in results if not r[1]]
    
    print(f"✅ 成功测试: {len(successful_tests)}/{len(results)}")
    for name, success, params in successful_tests:
        print(f"   {name}: {params:,} 参数")
    
    if failed_tests:
        print(f"\n❌ 失败测试: {len(failed_tests)}")
        for name, success, params in failed_tests:
            print(f"   {name}")
    
    # 4. 梯度测试
    if successful_tests:
        print(f"\n🔥 梯度测试...")
        try:
            # 选择第一个成功的模型进行梯度测试
            test_name = successful_tests[0][0]
            
            if test_name == "高采样方案":
                from final_improvements.high_sampling_approach import HighSamplingEdgeMatchingNet
                model = HighSamplingEdgeMatchingNet(segment_length=30, num_samples=5, feature_dim=64).to(device)
            elif test_name == "基础傅里叶方案":
                from final_improvements.fourier_approach import FourierBasedMatchingNet
                model = FourierBasedMatchingNet(max_points=1000, num_freqs=32, feature_dim=64).to(device)
            elif test_name == "混合傅里叶方案":
                from final_improvements.fourier_approach import HybridFourierNet
                model = HybridFourierNet(max_points=1000, num_freqs=32, feature_dim=64, num_samples=3).to(device)
            
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            
            labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
            
            optimizer.zero_grad()
            output = model(points1, points2)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            print(f"   ✅ 梯度测试成功: {test_name}")
            print(f"   损失值: {loss.item():.4f}")
            
        except Exception as e:
            print(f"   ❌ 梯度测试失败: {e}")
    
    return len(successful_tests), len(results)

if __name__ == "__main__":
    success_count, total_count = test_network_basic_functionality()
    print(f"\n🎯 总体结果: {success_count}/{total_count} 成功")