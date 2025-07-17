#!/usr/bin/env python3
"""
快速测试混合方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from hybrid_approach.hybrid_network import HybridEdgeMatchingNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def quick_test_hybrid():
    """快速测试混合方法"""
    print("🚀 混合方法快速测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl", 
        "dataset/test_set.pkl",
        batch_size=8,  # 小batch size
        max_points=1000,
        num_workers=0  # 避免多进程问题
    )
    
    # 测试不同配置
    configs = [
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
        print(f"   采样方法: {config['sample_method']}")
        print(f"   集成方法: {config['ensemble_method']}")
        
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
        
        # 训练几个epoch
        epochs = 5
        best_acc = 0.0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # 限制batch数量
                    break
                    
                points1 = batch['source_points'].to(device)
                points2 = batch['target_points'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                logits = model(points1, points2)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
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
                    if batch_idx >= 5:  # 限制batch数量
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
            best_acc = max(best_acc, val_acc)
            
            print(f"   Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}")
        
        results.append({
            'name': config['name'],
            'config': config,
            'best_acc': best_acc
        })
    
    # 总结结果
    print(f"\n📊 测试结果总结")
    print("=" * 50)
    
    # 按准确率排序
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['name']}: {result['best_acc']:.4f}")
    
    # 与baseline比较
    baseline_acc = 0.6095
    best_result = results[0]
    
    print(f"\n🏆 最佳结果: {best_result['name']}")
    print(f"   准确率: {best_result['best_acc']:.4f}")
    print(f"   Baseline: {baseline_acc:.4f}")
    
    improvement = best_result['best_acc'] - baseline_acc
    if improvement > 0:
        print(f"   提升: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
    else:
        print(f"   下降: {improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    results = quick_test_hybrid()