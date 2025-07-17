#!/usr/bin/env python3
"""
快速测试改进方案
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import time

from hybrid_approach.improved_approach import ImprovedEdgeMatchingNet
from simplified_approach.dataset_simple import create_simple_dataloaders

def quick_test():
    """快速测试改进方案"""
    print("🚀 快速测试改进方案")
    print("=" * 50)
    
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
    
    # 创建模型
    print("🧠 创建改进模型...")
    model = ImprovedEdgeMatchingNet(max_points=1000, feature_dim=128).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    epochs = 15
    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    
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
        
        print(f'   Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, '
              f'F1={f1:.4f}, AUC={auc:.4f}, Time={epoch_time:.1f}s')
    
    print(f"\n🏆 最佳结果:")
    print(f"   准确率: {best_acc:.4f}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   AUC: {best_auc:.4f}")
    
    # 与已知结果对比
    print(f"\n📊 与已知结果对比:")
    comparisons = [
        ('原始复杂网络', 0.5000),
        ('简化网络', 0.5985),
        ('final_approach', 0.6095),
        ('混合方法(单采样)', 0.5839),
        ('改进方案', best_acc)
    ]
    
    print("方法                    | 准确率")
    print("-" * 35)
    for method, acc in comparisons:
        if method == '改进方案':
            print(f"{method:<20} | {acc:.4f} 🏆")
        else:
            print(f"{method:<20} | {acc:.4f}")
    
    # 判断是否需要进一步改进
    baseline_acc = 0.6095  # final_approach
    improvement = best_acc - baseline_acc
    
    print(f"\n📈 改进分析:")
    print(f"   与final_approach对比: {improvement:+.4f} ({improvement/baseline_acc*100:+.1f}%)")
    
    if improvement > 0.01:
        print("   ✅ 显著改进，方案有效")
        return True, best_acc
    elif improvement > 0.005:
        print("   ⚠️ 小幅改进，可考虑进一步优化")
        return False, best_acc
    else:
        print("   ❌ 无明显改进，需要重新设计")
        return False, best_acc

if __name__ == "__main__":
    success, acc = quick_test()