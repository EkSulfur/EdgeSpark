"""
改进的EdgeSpark方法：引入PairingNet的关键技术
1. FocalLoss处理样本不平衡
2. 温度缩放的相似度计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import math
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from improved_dataset_loader import create_improved_dataloaders

class FocalLoss(nn.Module):
    """
    FocalLoss: 专门处理样本不平衡问题
    基于PairingNet的实现，但简化为二分类版本
    """
    def __init__(self, alpha=0.55, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits (batch_size, 1)
            targets: 真实标签 (batch_size, 1)
        """
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        p_t = torch.exp(-bce_loss)
        
        # 计算alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算focal权重
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # 应用focal权重
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EdgeShapeEncoder(nn.Module):
    """
    边缘形状编码器 - 保持原有设计
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

class ImprovedEdgeMatchingNet(nn.Module):
    """
    改进的边缘匹配网络：引入温度缩放
    """
    def __init__(self, max_points=1000, temperature=1.0):
        super().__init__()
        self.max_points = max_points
        self.temperature = temperature
        
        # 形状编码器
        self.shape_encoder = EdgeShapeEncoder(max_points)
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            # 输入：两个64维特征拼接 + 差值 + 温度缩放的相似度
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
        
    def get_temperature_scaled_similarity(self, features1, features2):
        """
        计算温度缩放的相似度
        Args:
            features1: (batch_size, 64)
            features2: (batch_size, 64)
        Returns:
            scaled_similarity: (batch_size, 1)
        """
        # 计算余弦相似度
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)
        
        # 点积相似度
        dot_similarity = torch.sum(features1_norm * features2_norm, dim=1, keepdim=True)
        
        # 温度缩放
        scaled_similarity = dot_similarity / self.temperature
        
        return scaled_similarity
        
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
        temp_sim = self.get_temperature_scaled_similarity(shape1, shape2)  # 温度缩放相似度
        
        # 拼接所有特征
        combined = torch.cat([shape1, shape2, diff, temp_sim], dim=1)  # (batch_size, 64*2+64+1)
        
        # 匹配预测
        match_logits = self.matching_net(combined)
        
        return match_logits

class ImprovedFinalTrainer:
    """改进的最终训练器：使用FocalLoss和温度缩放"""
    
    def __init__(self, temperature=1.0, alpha=0.55, gamma=2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        
        # 创建改进的模型
        self.model = ImprovedEdgeMatchingNet(
            max_points=1000, 
            temperature=temperature
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        
        # 使用FocalLoss
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        
        # 历史记录
        self.history = []
        
        print(f"✅ 改进训练器初始化完成:")
        print(f"   - 温度缩放: {temperature}")
        print(f"   - FocalLoss: α={alpha}, γ={gamma}")
        print(f"   - 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
            
            # 使用FocalLoss
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
        print(f"🚀 开始训练改进版本 (FocalLoss + 温度缩放)")
        
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
                torch.save(self.model.state_dict(), 'best_improved_model.pth')
            
            # 提前停止检查
            if epoch >= 10 and val_acc < 0.52:
                print("🔴 验证准确率过低，提前停止")
                break
        
        print(f"\n🎉 训练完成！最佳准确率: {best_acc:.4f}")
        return best_acc

def run_improved_experiment(sampling_strategy='ordered', temperature=1.0, alpha=0.55, gamma=2.0, epochs=25):
    """运行改进实验"""
    print(f"\n🔬 改进实验: 策略={sampling_strategy}, T={temperature}, α={alpha}, γ={gamma}")
    print("=" * 70)
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_improved_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl", 
        "dataset/test_set.pkl",
        batch_size=32,
        max_points=1000,
        num_workers=4,
        sampling_strategy=sampling_strategy
    )
    
    # 创建改进的训练器
    trainer = ImprovedFinalTrainer(temperature=temperature, alpha=alpha, gamma=gamma)
    
    # 开始训练
    best_acc = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 保存模型
    model_name = f'best_improved_model_{sampling_strategy}_T{temperature}_a{alpha}_g{gamma}.pth'
    if os.path.exists('best_improved_model.pth'):
        os.rename('best_improved_model.pth', model_name)
        print(f"💾 模型已保存为: {model_name}")
    
    return best_acc, trainer.history

def main():
    """主函数 - 运行改进实验"""
    print("🔥 EdgeSpark改进实验 - FocalLoss + 温度缩放")
    print("=" * 70)
    
    # 实验配置
    experiments = [
        {'strategy': 'ordered', 'temperature': 1.0, 'alpha': 0.55, 'gamma': 2.0},
        {'strategy': 'ordered', 'temperature': 0.5, 'alpha': 0.55, 'gamma': 2.0},
        {'strategy': 'ordered', 'temperature': 2.0, 'alpha': 0.55, 'gamma': 2.0},
        {'strategy': 'ordered', 'temperature': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        {'strategy': 'ordered', 'temperature': 1.0, 'alpha': 0.75, 'gamma': 2.0},
    ]
    
    results = {}
    
    # 运行每个实验
    for i, exp in enumerate(experiments):
        try:
            print(f"\n🔬 实验 {i+1}/{len(experiments)}")
            
            best_acc, history = run_improved_experiment(
                sampling_strategy=exp['strategy'],
                temperature=exp['temperature'],
                alpha=exp['alpha'],
                gamma=exp['gamma'],
                epochs=25
            )
            
            exp_name = f"{exp['strategy']}_T{exp['temperature']}_a{exp['alpha']}_g{exp['gamma']}"
            results[exp_name] = {
                'best_acc': best_acc,
                'history': history,
                'params': exp
            }
            
            print(f"✅ 实验完成: {exp_name} = {best_acc:.4f}")
            
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            exp_name = f"{exp['strategy']}_T{exp['temperature']}_a{exp['alpha']}_g{exp['gamma']}"
            results[exp_name] = {'best_acc': 0.0, 'error': str(e)}
    
    # 结果分析
    print(f"\n" + "=" * 70)
    print("📈 改进实验结果")
    print("=" * 70)
    
    best_exp = None
    best_score = 0.0
    
    for exp_name, result in results.items():
        if 'error' not in result:
            acc = result['best_acc']
            params = result['params']
            print(f"{exp_name:>25}: {acc:.4f} (T={params['temperature']}, α={params['alpha']}, γ={params['gamma']})")
            if acc > best_score:
                best_score = acc
                best_exp = exp_name
        else:
            print(f"{exp_name:>25}: 失败 ({result['error']})")
    
    print(f"\n🏆 最佳实验: {best_exp} (准确率: {best_score:.4f})")
    
    # 保存实验结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_experiment_results_{timestamp}.json"
    
    # 转换为可序列化格式
    serializable_results = {}
    for exp_name, result in results.items():
        if 'history' in result:
            serializable_results[exp_name] = {
                'best_acc': result['best_acc'],
                'params': result['params'],
                'final_epoch': len(result['history']),
                'final_val_acc': result['history'][-1]['val_acc'] if result['history'] else 0.0
            }
        else:
            serializable_results[exp_name] = result
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📁 实验结果已保存到: {results_file}")
    
    # 改进效果分析
    print(f"\n=== 改进效果分析 ===")
    if best_score > 0.8:
        print("🎉 显著改进！FocalLoss + 温度缩放效果显著")
    elif best_score > 0.7:
        print("✅ 明显改进！技术有效")
    elif best_score > 0.6:
        print("⚠️  小幅改进，需要进一步优化")
    else:
        print("❌ 改进效果不明显")
        print("💡 建议：")
        print("   1. 调整温度参数范围")
        print("   2. 优化FocalLoss参数")
        print("   3. 改进特征编码器")
        print("   4. 增加数据增强")
    
    return best_score

if __name__ == "__main__":
    result = main()