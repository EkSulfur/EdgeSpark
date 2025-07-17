#!/usr/bin/env python3
"""
完整的混合方法测试
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import json
from datetime import datetime

from hybrid_approach.hybrid_network import HybridEdgeMatchingNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def full_test_hybrid():
    """完整测试混合方法"""
    print("🚀 混合方法完整测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl", 
        "dataset/test_set.pkl",
        batch_size=16,
        max_points=1000,
        num_workers=4
    )
    
    # 测试配置
    configs = [
        {
            'name': 'Baseline-单次采样',
            'num_samples': 1,
            'sample_method': 'diversified',
            'ensemble_method': 'simple_average'
        },
        {
            'name': '3采样-多样化-简单平均',
            'num_samples': 3,
            'sample_method': 'diversified',
            'ensemble_method': 'simple_average'
        },
        {
            'name': '5采样-多样化-加权平均',
            'num_samples': 5,
            'sample_method': 'diversified',
            'ensemble_method': 'weighted_average'
        },
        {
            'name': '3采样-随机-置信度加权',
            'num_samples': 3,
            'sample_method': 'random',
            'ensemble_method': 'confidence_weighted'
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 测试配置: {config['name']}")
        print(f"   采样次数: {config['num_samples']}")
        
        # 创建模型
        model = HybridEdgeMatchingNet(
            max_points=1000,
            num_samples=config['num_samples'],
            sample_method=config['sample_method'],
            ensemble_method=config['ensemble_method']
        ).to(device)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练
        epochs = 15
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
                points1 = batch['source_points'].to(device)
                points2 = batch['target_points'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
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
                
                if batch_idx % 50 == 0:
                    print(f'     Batch {batch_idx:3d}/{len(train_loader):3d} | Loss: {loss.item():.4f}')
            
            train_acc = accuracy_score(train_labels, train_preds)
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    points1 = batch['source_points'].to(device)
                    points2 = batch['target_points'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(points1, points2)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
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
            
            print(f'   Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, '
                  f'F1={f1:.4f}, AUC={auc:.4f}, Time={epoch_time:.1f}s')
        
        results.append({
            'name': config['name'],
            'config': config,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_auc': best_auc,
            'training_time': training_time,
            'params': sum(p.numel() for p in model.parameters())
        })
        
        print(f'   ✅ 完成: 最佳准确率={best_acc:.4f}, 训练时间={training_time:.1f}s')
    
    # 总结结果
    print(f"\n📊 完整测试结果总结")
    print("=" * 80)
    
    # 按准确率排序
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'排名':<4} {'方法':<20} {'准确率':<8} {'F1':<8} {'AUC':<8} {'训练时间':<10} {'参数量':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['name']:<20} {result['best_acc']:.4f}   {result['best_f1']:.4f}   "
              f"{result['best_auc']:.4f}   {result['training_time']:.1f}s     {result['params']:,}")
    
    # 分析结果
    print(f"\n📈 结果分析")
    print("=" * 50)
    
    baseline_acc = 0.6095  # final_approach准确率
    best_result = results[0]
    
    print(f"🏆 最佳方法: {best_result['name']}")
    print(f"   准确率: {best_result['best_acc']:.4f}")
    print(f"   F1-Score: {best_result['best_f1']:.4f}")
    print(f"   AUC: {best_result['best_auc']:.4f}")
    print(f"   训练时间: {best_result['training_time']:.1f}s")
    
    improvement = best_result['best_acc'] - baseline_acc
    print(f"\n📊 与baseline (final_approach: {baseline_acc:.4f}) 比较:")
    if improvement > 0:
        print(f"   ✅ 提升: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
    else:
        print(f"   ❌ 下降: {improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
    
    # 分析多采样的效果
    single_sample = next((r for r in results if r['config']['num_samples'] == 1), None)
    multi_sample = [r for r in results if r['config']['num_samples'] > 1]
    
    if single_sample and multi_sample:
        print(f"\n🔍 多采样效果分析:")
        print(f"   单次采样: {single_sample['best_acc']:.4f}")
        
        for result in multi_sample:
            samples = result['config']['num_samples']
            diff = result['best_acc'] - single_sample['best_acc']
            time_ratio = result['training_time'] / single_sample['training_time']
            
            print(f"   {samples}次采样: {result['best_acc']:.4f} "
                  f"({'+' if diff > 0 else ''}{diff:.4f}, "
                  f"{time_ratio:.1f}x时间)")
    
    # 保存结果
    save_path = f'hybrid_experiments/full_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 结果已保存到: {save_path}")
    
    return results

if __name__ == "__main__":
    results = full_test_hybrid()