#!/usr/bin/env python3
"""
快速评估最终改进方案
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from sklearn.metrics import accuracy_score

# 添加路径
sys.path.append('/home/eksulfur/EdgeSpark')

from final_improvements.high_sampling_approach import HighSamplingEdgeMatchingNet
from final_improvements.fourier_approach import FourierBasedMatchingNet, HybridFourierNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def quick_evaluation():
    """快速评估改进方案"""
    print("🚀 EdgeSpark 最终改进方案快速评估")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器（小批量测试）
    print("\n📚 准备数据...")
    try:
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            "dataset/train_set.pkl",
            "dataset/valid_set.pkl",
            "dataset/test_set.pkl",
            batch_size=8,  # 小批量
            max_points=500,  # 较少点数
            num_workers=1
        )
        print(f"   数据加载成功")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return []
    
    # 测试配置
    models = [
        {
            'name': '高采样方案(5采样)',
            'model': HighSamplingEdgeMatchingNet(segment_length=30, num_samples=5, feature_dim=64)
        },
        {
            'name': '基础傅里叶方案',
            'model': FourierBasedMatchingNet(max_points=500, num_freqs=32, feature_dim=64)
        },
        {
            'name': '混合傅里叶方案', 
            'model': HybridFourierNet(max_points=500, num_freqs=32, feature_dim=64, num_samples=3)
        }
    ]
    
    results = []
    
    for model_config in models:
        print(f"\n🧪 测试: {model_config['name']}")
        print("-" * 40)
        
        model = model_config['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # 快速训练（5轮）
        best_acc = 0.0
        training_time = 0.0
        
        for epoch in range(5):
            epoch_start = time.time()
            
            # 训练
            model.train()
            train_preds = []
            train_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 20:  # 限制批次
                    break
                    
                points1 = batch['source_points'].to(device)
                points2 = batch['target_points'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                logits = model(points1, points2)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_acc = accuracy_score(train_labels, train_preds)
            
            # 验证
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:  # 限制验证批次
                        break
                        
                    points1 = batch['source_points'].to(device)
                    points2 = batch['target_points'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(points1, points2)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            
            if val_acc > best_acc:
                best_acc = val_acc
                
            epoch_time = time.time() - epoch_start
            training_time += epoch_time
            
            print(f"   Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}, Time={epoch_time:.1f}s")
        
        results.append({
            'name': model_config['name'],
            'best_acc': best_acc,
            'training_time': training_time,
            'params': sum(p.numel() for p in model.parameters())
        })
        
        print(f"   ✅ 最佳准确率: {best_acc:.4f}")
    
    # 总结结果
    print(f"\n📊 快速评估结果总结")
    print("=" * 60)
    
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'排名':<4} {'方法':<20} {'准确率':<8} {'参数量':<10} {'训练时间':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['name']:<20} {result['best_acc']:.4f}   "
              f"{result['params']:,}   {result['training_time']:.1f}s")
    
    # 与历史基准对比
    print(f"\n📈 与历史基准对比")
    print("-" * 40)
    
    baselines = [
        ('final_approach', 0.6095),
        ('混合方法(单采样)', 0.5839),
        ('改进方案(特征工程)', 0.6100)
    ]
    
    for method, acc in baselines:
        print(f"   {method}: {acc:.4f}")
    
    if results:
        best_result = results[0]
        print(f"   🏆 {best_result['name']}: {best_result['best_acc']:.4f}")
        
        baseline_acc = 0.6095
        improvement = best_result['best_acc'] - baseline_acc
        print(f"\n💡 改进分析:")
        print(f"   与final_approach对比: {improvement:+.4f} ({improvement/baseline_acc*100:+.1f}%)")
        
        if improvement > 0.02:
            print("   ✅ 显著改进 - 用户建议非常有效!")
        elif improvement > 0.01:
            print("   ✅ 明显改进 - 用户建议有效")
        elif improvement > 0.005:
            print("   ⚠️ 小幅改进 - 方向正确") 
        else:
            print("   ❌ 无明显改进 - 需要进一步调优")
    
    return results

if __name__ == "__main__":
    results = quick_evaluation()