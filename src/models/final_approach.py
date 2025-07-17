"""
最终方法：基于边缘形状特征的网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from improved_dataset_loader import create_improved_dataloaders

class EdgeShapeEncoder(nn.Module):
    """
    边缘形状编码器
    专门设计用于捕捉边缘形状特征
    """
    def __init__(self, max_points=1000):
        super().__init__()
        self.max_points = max_points
        
        # 1. 局部形状特征提取
        self.local_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 2. 全局形状特征提取
        self.global_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 3. 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. 最终投影
        self.final_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
    def forward(self, points):
        """
        编码边缘形状
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, 64)
        """
        # 转换维度
        x = points.transpose(1, 2)  # (batch_size, 2, num_points)
        
        # 局部特征提取
        local_features = self.local_conv(x)  # (batch_size, 256, num_points)
        
        # 全局特征提取
        global_features = self.global_conv(local_features)  # (batch_size, 256, num_points)
        
        # 自适应池化
        pooled = self.adaptive_pool(global_features).squeeze(-1)  # (batch_size, 256)
        
        # 最终投影
        final_features = self.final_proj(pooled)  # (batch_size, 64)
        
        return final_features

class EdgeMatchingNet(nn.Module):
    """
    边缘匹配网络
    """
    def __init__(self, max_points=1000):
        super().__init__()
        self.max_points = max_points
        
        # 形状编码器
        self.shape_encoder = EdgeShapeEncoder(max_points)
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            # 输入：两个64维特征拼接 + 差值 + 点积
            nn.Linear(64 * 2 + 64 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, points1, points2):
        """
        前向传播
        Args:
            points1: (batch_size, num_points1, 2)
            points2: (batch_size, num_points2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        # 形状编码
        shape1 = self.shape_encoder(points1)  # (batch_size, 64)
        shape2 = self.shape_encoder(points2)  # (batch_size, 64)
        
        # 特征组合
        diff = shape1 - shape2  # 差值特征
        dot = torch.sum(shape1 * shape2, dim=1, keepdim=True)  # 点积相似度
        
        # 拼接所有特征
        combined = torch.cat([shape1, shape2, diff, dot], dim=1)  # (batch_size, 64*2+64+1)
        
        # 匹配预测
        match_logits = self.matching_net(combined)
        
        return match_logits

class FinalTrainer:
    """最终训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = EdgeMatchingNet(max_points=1000).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 历史记录
        self.history = []
        
    def train_epoch(self, loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(loader):
            points1 = batch['source_points'].to(self.device)
            points2 = batch['target_points'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(points1, points2)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f'    Batch {batch_idx:3d}/{len(loader):3d} | Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, acc
    
    def validate_epoch(self, loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                points1 = batch['source_points'].to(self.device)
                points2 = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(points1, points2)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            auc = roc_auc_score(all_labels, all_probs)
        except:
            precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5
        
        return avg_loss, acc, precision, recall, f1, auc
    
    def train(self, train_loader, val_loader, epochs=30):
        """训练循环"""
        print(f"🚀 开始训练最终版本")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate_epoch(val_loader)
            
            # 学习率更新
            self.scheduler.step()
            
            # 记录
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 显示结果
            print(f"📈 训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"📊 验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            print(f"📚 学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"  💾 新的最佳准确率: {best_acc:.4f}")
                
                # 保存模型
                torch.save(self.model.state_dict(), 'best_final_model.pth')
            
            # 提前停止检查
            if epoch >= 10 and val_acc < 0.52:
                print("🔴 验证准确率过低，提前停止")
                break
        
        print(f"\n🎉 训练完成！最佳准确率: {best_acc:.4f}")
        return best_acc

def run_experiment(sampling_strategy='ordered', epochs=25):
    """运行单个实验"""
    print(f"\n🔬 实验: 采样策略 = {sampling_strategy}")
    print("=" * 60)
    
    # 创建数据加载器
    print("📚 创建改进版数据加载器...")
    train_loader, val_loader, test_loader = create_improved_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl", 
        "dataset/test_set.pkl",
        batch_size=32,
        max_points=1000,
        num_workers=4,
        sampling_strategy=sampling_strategy
    )
    
    # 创建训练器
    trainer = FinalTrainer()
    
    # 开始训练
    best_acc = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 保存模型（加上策略名称）
    if os.path.exists('best_final_model.pth'):
        os.rename('best_final_model.pth', f'best_final_model_{sampling_strategy}.pth')
        print(f"💾 模型已保存为: best_final_model_{sampling_strategy}.pth")
    
    return best_acc, trainer.history

def main():
    """主函数 - 运行多个实验对比"""
    print("🔥 EdgeSpark最终尝试 - 改进版DataLoader实验")
    print("=" * 70)
    
    # 实验配置
    strategies = ['ordered', 'random', 'padding']
    results = {}
    
    # 运行每个策略的实验
    for strategy in strategies:
        try:
            best_acc, history = run_experiment(strategy, epochs=25)
            results[strategy] = {
                'best_acc': best_acc,
                'history': history
            }
            
            print(f"\n📊 {strategy}策略结果: 最佳准确率 = {best_acc:.4f}")
            
        except Exception as e:
            print(f"❌ {strategy}策略实验失败: {e}")
            results[strategy] = {'best_acc': 0.0, 'error': str(e)}
    
    # 结果对比分析
    print(f"\n" + "=" * 70)
    print("📈 实验结果对比")
    print("=" * 70)
    
    best_strategy = None
    best_score = 0.0
    
    for strategy, result in results.items():
        if 'error' not in result:
            acc = result['best_acc']
            print(f"{strategy:>10}策略: {acc:.4f}")
            if acc > best_score:
                best_score = acc
                best_strategy = strategy
        else:
            print(f"{strategy:>10}策略: 失败 ({result['error']})")
    
    print(f"\n🏆 最佳策略: {best_strategy} (准确率: {best_score:.4f})")
    
    # 保存实验结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_dataloader_results_{timestamp}.json"
    
    # 转换history为可序列化格式
    serializable_results = {}
    for strategy, result in results.items():
        if 'history' in result:
            serializable_results[strategy] = {
                'best_acc': result['best_acc'],
                'final_epoch': len(result['history']),
                'final_val_acc': result['history'][-1]['val_acc'] if result['history'] else 0.0
            }
        else:
            serializable_results[strategy] = result
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📁 实验结果已保存到: {results_file}")
    
    # 结果分析
    print(f"\n=== 最终分析 ===")
    if best_score > 0.7:
        print("🎉 成功！改进的数据加载器显著提升了性能")
    elif best_score > 0.6:
        print("⚠️  部分成功，改进的数据加载器有一定效果")
    else:
        print("❌ 需要进一步优化数据处理策略")
        print("💡 建议：")
        print("   1. 尝试其他采样策略")
        print("   2. 调整max_points参数")
        print("   3. 增强数据增强策略")
        print("   4. 检查边缘点云质量")
    
    return best_score

if __name__ == "__main__":
    result = main()