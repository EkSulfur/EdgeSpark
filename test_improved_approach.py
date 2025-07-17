#!/usr/bin/env python3
"""
测试改进方案
专注于特征工程而非多采样
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from hybrid_approach.improved_approach import ImprovedEdgeMatchingNet
# 为了对比，我们创建一个简单的基线网络
class BaselineNet(nn.Module):
    def __init__(self, max_points=1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 + 64 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, points1, points2):
        # 全局平均池化
        feat1 = torch.mean(self.encoder(points1), dim=1)
        feat2 = torch.mean(self.encoder(points2), dim=1)
        
        diff = feat1 - feat2
        dot = torch.sum(feat1 * feat2, dim=1, keepdim=True)
        combined = torch.cat([feat1, feat2, diff, dot], dim=1)
        return self.classifier(combined)
from simplified_approach.dataset_simple import create_simple_dataloaders

def test_improved_approach():
    """测试改进方案"""
    print("🚀 测试改进方案")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=32,
        max_points=1000,
        num_workers=4
    )
    
    # 测试模型
    models = [
        {
            'name': 'Simple Baseline',
            'model': BaselineNet(max_points=1000)
        },
        {
            'name': 'Improved Approach',
            'model': ImprovedEdgeMatchingNet(max_points=1000, feature_dim=128)
        }
    ]
    
    results = []
    
    for model_config in models:
        print(f"\n🧪 测试模型: {model_config['name']}")
        print("=" * 40)
        
        model = model_config['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # 训练
        epochs = 20
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
            
            scheduler.step()
            
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
            
            # 提前停止
            if epoch - ([i for i, h in enumerate(results) if h['best_acc'] == best_acc] or [0])[-1] > 5:
                print(f'   Early stopping at epoch {epoch+1}')
                break
        
        results.append({
            'name': model_config['name'],
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_auc': best_auc,
            'training_time': training_time,
            'params': sum(p.numel() for p in model.parameters()),
            'avg_epoch_time': training_time / (epoch + 1)
        })
        
        print(f'   ✅ 完成: 最佳准确率={best_acc:.4f}, 总训练时间={training_time:.1f}s')
    
    # 总结结果
    print(f"\n📊 改进方案测试结果")
    print("=" * 80)
    
    # 按准确率排序
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'排名':<4} {'方法':<25} {'准确率':<8} {'F1':<8} {'AUC':<8} {'参数量':<10} {'平均时间':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['name']:<25} {result['best_acc']:.4f}   {result['best_f1']:.4f}   "
              f"{result['best_auc']:.4f}   {result['params']:,}   {result['avg_epoch_time']:.1f}s")
    
    # 详细分析
    print(f"\n📈 详细分析")
    print("=" * 50)
    
    if len(results) >= 2:
        baseline = results[1] if results[0]['name'] == 'Improved Approach' else results[0]
        improved = results[0] if results[0]['name'] == 'Improved Approach' else results[1]
        
        print(f"🏆 最佳方法: {results[0]['name']}")
        print(f"   准确率: {results[0]['best_acc']:.4f}")
        print(f"   F1-Score: {results[0]['best_f1']:.4f}")
        print(f"   AUC: {results[0]['best_auc']:.4f}")
        
        acc_diff = improved['best_acc'] - baseline['best_acc']
        f1_diff = improved['best_f1'] - baseline['best_f1']
        auc_diff = improved['best_auc'] - baseline['best_auc']
        time_diff = improved['avg_epoch_time'] - baseline['avg_epoch_time']
        
        print(f"\n📊 改进方案 vs Baseline:")
        print(f"   准确率差异: {acc_diff:+.4f} ({acc_diff/baseline['best_acc']*100:+.1f}%)")
        print(f"   F1差异: {f1_diff:+.4f} ({f1_diff/baseline['best_f1']*100:+.1f}%)")
        print(f"   AUC差异: {auc_diff:+.4f} ({auc_diff/baseline['best_auc']*100:+.1f}%)")
        print(f"   训练时间差异: {time_diff:+.1f}s/epoch")
        
        # 综合评估
        print(f"\n🎯 综合评估:")
        if acc_diff > 0.01:
            print("   ✅ 准确率显著提升")
        elif acc_diff > 0.005:
            print("   ⚠️ 准确率小幅提升")
        else:
            print("   ❌ 准确率无明显提升")
        
        if time_diff < 5:
            print("   ✅ 训练时间开销可接受")
        else:
            print("   ⚠️ 训练时间开销较大")
    
    # 与其他方法对比
    print(f"\n📋 与其他方法对比:")
    comparisons = [
        ('原始复杂网络', 0.5000),
        ('简化网络', 0.5985),
        ('final_approach', 0.6095),
        ('混合方法(单采样)', 0.5839),
        ('改进方案', results[0]['best_acc'])
    ]
    
    for method, acc in comparisons:
        if method == '改进方案':
            print(f"   🏆 {method}: {acc:.4f}")
        else:
            print(f"   📊 {method}: {acc:.4f}")
    
    return results

if __name__ == "__main__":
    results = test_improved_approach()