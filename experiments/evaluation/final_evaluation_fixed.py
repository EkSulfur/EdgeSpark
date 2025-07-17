#!/usr/bin/env python3
"""
修复版最终评估脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# 添加路径
sys.path.append('/home/eksulfur/EdgeSpark')

from final_improvements.high_sampling_approach import HighSamplingEdgeMatchingNet
from final_improvements.fourier_approach import FourierBasedMatchingNet, HybridFourierNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def final_evaluation():
    """最终评估方案"""
    print("🎯 EdgeSpark 最终改进方案评估")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器
    print("\n📚 准备数据...")
    try:
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            "dataset/train_set.pkl",
            "dataset/valid_set.pkl",
            "dataset/test_set.pkl",
            batch_size=16,
            max_points=1000,
            num_workers=2
        )
        print(f"   训练样本: {len(train_loader.dataset)}")
        print(f"   验证样本: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return []
    
    # 测试配置
    models = [
        {
            'name': '高采样方案(10采样)',
            'model': HighSamplingEdgeMatchingNet(segment_length=50, num_samples=10, feature_dim=128),
            'lr': 0.0005
        },
        {
            'name': '基础傅里叶方案',
            'model': FourierBasedMatchingNet(max_points=1000, num_freqs=64, feature_dim=128),
            'lr': 0.001
        },
        {
            'name': '混合傅里叶方案',
            'model': HybridFourierNet(max_points=1000, num_freqs=64, feature_dim=128, num_samples=5),
            'lr': 0.0008
        }
    ]
    
    results = []
    
    for model_config in models:
        print(f"\n🧪 测试: {model_config['name']}")
        print("-" * 40)
        
        model = model_config['model'].to(device)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=model_config['lr'], 
            weight_decay=1e-3
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        
        # 训练参数
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        training_time = 0.0
        
        for epoch in range(12):  # 适中的轮数
            epoch_start = time.time()
            
            # 训练
            model.train()
            train_preds = []
            train_labels = []
            
            # 限制训练批次数量
            max_train_batches = min(80, len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_train_batches:
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
                
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
            
            train_acc = accuracy_score(train_labels, train_preds)
            
            # 验证（使用完整验证集）
            model.eval()
            val_preds = []
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    points1 = batch['source_points'].to(device)
                    points2 = batch['target_points'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(points1, points2)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            
            # 计算其他指标
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels, val_preds, average='binary', zero_division=0
                )
                auc = roc_auc_score(val_labels, val_probs)
            except:
                precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5
            
            # 更新最佳结果
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = f1
                best_auc = auc
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            training_time += epoch_time
            
            print(f"   Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
                  f"F1={f1:.4f}, AUC={auc:.4f}, Time={epoch_time:.1f}s")
        
        results.append({
            'name': model_config['name'],
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_auc': best_auc,
            'training_time': training_time,
            'params': sum(p.numel() for p in model.parameters())
        })
        
        print(f"   ✅ 最佳结果: 准确率={best_acc:.4f}, F1={best_f1:.4f}")
    
    # 总结结果
    print(f"\n📊 最终评估结果")
    print("=" * 70)
    
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'排名':<4} {'方法':<20} {'准确率':<8} {'F1':<8} {'AUC':<8} {'参数量':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['name']:<20} {result['best_acc']:.4f}   "
              f"{result['best_f1']:.4f}   {result['best_auc']:.4f}   {result['params']:,}")
    
    # 与历史基准对比
    print(f"\n📈 与历史基准对比")
    print("-" * 50)
    
    baselines = [
        ('final_approach (历史最佳)', 0.6095),
        ('混合方法(单采样)', 0.5839),
        ('简化网络', 0.5985)
    ]
    
    for method, acc in baselines:
        print(f"   {method}: {acc:.4f}")
    
    if results:
        best_result = results[0]
        print(f"   🏆 {best_result['name']}: {best_result['best_acc']:.4f}")
        
        # 改进分析
        baseline_acc = 0.6095
        improvement = best_result['best_acc'] - baseline_acc
        
        print(f"\n💡 改进分析")
        print("-" * 40)
        print(f"最佳方法: {best_result['name']}")
        print(f"准确率: {best_result['best_acc']:.4f}")
        print(f"F1-Score: {best_result['best_f1']:.4f}")
        print(f"AUC: {best_result['best_auc']:.4f}")
        print(f"与final_approach对比: {improvement:+.4f} ({improvement/baseline_acc*100:+.1f}%)")
        
        # 评估用户建议的有效性
        if improvement > 0.015:
            success_level = "显著成功"
            print("✅ 显著改进 - 用户建议非常有效!")
        elif improvement > 0.008:
            success_level = "成功"
            print("✅ 明显改进 - 用户建议有效")
        elif improvement > 0.003:
            success_level = "部分成功"
            print("⚠️ 小幅改进 - 方向正确")
        else:
            success_level = "需要改进"
            print("❌ 无明显改进 - 需要重新考虑")
        
        results[0]['success_level'] = success_level
        results[0]['improvement'] = improvement
    
    return results

if __name__ == "__main__":
    results = final_evaluation()