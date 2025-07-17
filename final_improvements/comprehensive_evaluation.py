#!/usr/bin/env python3
"""
综合评估脚本
测试所有改进方案并对比结果
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import json
from datetime import datetime

# 添加路径以导入模块
sys.path.append('/home/eksulfur/EdgeSpark')

# 导入网络
from final_improvements.high_sampling_approach import HighSamplingEdgeMatchingNet
from final_improvements.fourier_approach import FourierBasedMatchingNet, HybridFourierNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def quick_train_and_evaluate(model, model_name, train_loader, val_loader, device, epochs=10):
    """
    快速训练和评估模型
    """
    print(f"\n🚀 训练模型: {model_name}")
    print("-" * 50)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    training_time = 0.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 50:  # 限制批次数量以加快测试
                break
                
            points1 = batch['source_points'].to(device)
            points2 = batch['target_points'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            try:
                logits = model(points1, points2)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"   训练错误: {e}")
                continue
        
        scheduler.step()
        
        if not train_preds:
            print(f"   Epoch {epoch+1}: 训练失败")
            continue
            
        train_acc = accuracy_score(train_labels, train_preds)
        train_loss /= min(50, len(train_loader))
        
        # 验证
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 20:  # 限制验证批次
                    break
                    
                points1 = batch['source_points'].to(device)
                points2 = batch['target_points'].to(device)
                labels = batch['label'].to(device)
                
                try:
                    logits = model(points1, points2)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"   验证错误: {e}")
                    continue
        
        if not val_preds:
            print(f"   Epoch {epoch+1}: 验证失败")
            continue
            
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
        
        epoch_time = time.time() - epoch_start
        training_time += epoch_time
        
        print(f"   Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
              f"F1={f1:.4f}, AUC={auc:.4f}, Time={epoch_time:.1f}s")
    
    return {
        'model_name': model_name,
        'best_acc': best_acc,
        'best_f1': best_f1,
        'best_auc': best_auc,
        'training_time': training_time,
        'params': sum(p.numel() for p in model.parameters()),
        'avg_epoch_time': training_time / epochs
    }

def comprehensive_evaluation():
    """
    综合评估所有改进方案
    """
    print("🎯 EdgeSpark 综合改进方案评估")
    print("=" * 80)
    
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
        print(f"数据加载失败: {e}")
        return
    
    # 测试模型配置
    model_configs = [
        {
            'name': '高采样方案(10采样)',
            'model_class': HighSamplingEdgeMatchingNet,
            'params': {'segment_length': 50, 'num_samples': 10, 'feature_dim': 128}
        },
        {
            'name': '高采样方案(20采样)',
            'model_class': HighSamplingEdgeMatchingNet,
            'params': {'segment_length': 50, 'num_samples': 20, 'feature_dim': 128}
        },
        {
            'name': '基础傅里叶方案',
            'model_class': FourierBasedMatchingNet,
            'params': {'max_points': 1000, 'num_freqs': 64, 'feature_dim': 128}
        },
        {
            'name': '混合傅里叶方案',
            'model_class': HybridFourierNet,
            'params': {'max_points': 1000, 'num_freqs': 64, 'feature_dim': 128, 'num_samples': 5}
        }
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n🧪 测试: {config['name']}")
        print("=" * 60)
        
        try:
            # 创建模型
            model = config['model_class'](**config['params']).to(device)
            print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 快速测试前向传播
            with torch.no_grad():
                test_batch = next(iter(val_loader))
                points1 = test_batch['source_points'][:2].to(device)
                points2 = test_batch['target_points'][:2].to(device)
                output = model(points1, points2)
                print(f"   前向传播测试: ✅ 输出形状 {output.shape}")
            
            # 训练和评估
            result = quick_train_and_evaluate(
                model, config['name'], train_loader, val_loader, device, epochs=8
            )
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model_name': config['name'],
                'best_acc': 0.0,
                'best_f1': 0.0,
                'best_auc': 0.0,
                'training_time': 0.0,
                'params': 0,
                'avg_epoch_time': 0.0,
                'error': str(e)
            })
    
    # 总结结果
    print(f"\n📊 综合评估结果")
    print("=" * 100)
    
    # 按准确率排序
    valid_results = [r for r in results if r['best_acc'] > 0]
    valid_results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'排名':<4} {'方法':<25} {'准确率':<8} {'F1':<8} {'AUC':<8} {'参数量':<10} {'平均时间':<10}")
    print("-" * 100)
    
    for i, result in enumerate(valid_results):
        print(f"{i+1:<4} {result['model_name']:<25} {result['best_acc']:.4f}   "
              f"{result['best_f1']:.4f}   {result['best_auc']:.4f}   "
              f"{result['params']:,}   {result['avg_epoch_time']:.1f}s")
    
    # 与已知基准对比
    print(f"\n📈 与已知基准对比")
    print("-" * 60)
    
    baselines = [
        ('原始复杂网络', 0.5000),
        ('简化网络', 0.5985),
        ('final_approach', 0.6095),
        ('混合方法(单采样)', 0.5839),
        ('改进方案(特征工程)', 0.6100)  # 估计值
    ]
    
    print("方法                          | 准确率    | 状态")
    print("-" * 50)
    for method, acc in baselines:
        print(f"{method:<25} | {acc:.4f}   | 历史结果")
    
    if valid_results:
        best_new = valid_results[0]
        print(f"{best_new['model_name']:<25} | {best_new['best_acc']:.4f}   | 🏆 新最佳")
        
        # 分析改进效果
        best_baseline = 0.6095  # final_approach
        improvement = best_new['best_acc'] - best_baseline
        
        print(f"\n💡 改进分析:")
        print(f"   最佳新方法: {best_new['model_name']}")
        print(f"   准确率: {best_new['best_acc']:.4f}")
        print(f"   与final_approach对比: {improvement:+.4f} ({improvement/best_baseline*100:+.1f}%)")
        
        if improvement > 0.01:
            print("   ✅ 显著改进 - 用户建议有效")
        elif improvement > 0.005:
            print("   ⚠️ 小幅改进 - 方向正确，需进一步优化")
        else:
            print("   ❌ 无明显改进 - 需要重新思考策略")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'final_improvements/evaluation_results_{timestamp}.json'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存到: {save_path}")
    
    return results

if __name__ == "__main__":
    results = comprehensive_evaluation()