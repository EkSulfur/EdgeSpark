"""
增强特征提取器的训练器
使用几何特征和对比学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from improved_dataset_loader import create_improved_dataloaders
from enhanced_feature_extractor import EnhancedFragmentMatcher, ContrastiveLoss

class EnhancedTrainer:
    """增强特征提取器训练器"""
    
    def __init__(self, use_contrastive=True, contrastive_weight=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = EnhancedFragmentMatcher(max_points=1000).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(margin=0.5, temperature=0.1)
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        # 历史记录
        self.history = []
        
        print(f"🚀 增强特征提取器初始化完成")
        print(f"📊 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 使用对比学习: {use_contrastive}")
        
    def train_epoch(self, loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_bce_loss = 0.0
        total_contrastive_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(loader):
            points1 = batch['source_points'].to(self.device)
            points2 = batch['target_points'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits, features1, features2 = self.model(points1, points2)
            
            # BCE损失
            bce_loss = self.bce_loss(logits, labels)
            
            # 对比损失
            contrastive_loss = 0
            if self.use_contrastive:
                contrastive_loss = self.contrastive_loss(features1, features2, labels.squeeze())
            
            # 总损失
            total_batch_loss = bce_loss + self.contrastive_weight * contrastive_loss
            
            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            total_bce_loss += bce_loss.item()
            if self.use_contrastive:
                total_contrastive_loss += contrastive_loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 20 == 0:
                if self.use_contrastive:
                    print(f'    Batch {batch_idx:3d}/{len(loader):3d} | '
                          f'Loss: {total_batch_loss.item():.4f} '
                          f'(BCE: {bce_loss.item():.4f}, Cont: {contrastive_loss.item():.4f})')
                else:
                    print(f'    Batch {batch_idx:3d}/{len(loader):3d} | Loss: {total_batch_loss.item():.4f}')
        
        avg_loss = total_loss / len(loader)
        avg_bce_loss = total_bce_loss / len(loader)
        avg_contrastive_loss = total_contrastive_loss / len(loader) if self.use_contrastive else 0
        acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_bce_loss, avg_contrastive_loss, acc
    
    def validate_epoch(self, loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        all_features1 = []
        all_features2 = []
        
        with torch.no_grad():
            for batch in loader:
                points1 = batch['source_points'].to(self.device)
                points2 = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, features1, features2 = self.model(points1, points2)
                loss = self.bce_loss(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features1.extend(features1.cpu().numpy())
                all_features2.extend(features2.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            auc = roc_auc_score(all_labels, all_probs)
        except:
            precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5
        
        # 计算特征可分离性
        separability = self.compute_feature_separability(all_features1, all_features2, all_labels)
        
        return avg_loss, acc, precision, recall, f1, auc, separability, all_features1, all_features2, all_labels
    
    def compute_feature_separability(self, features1, features2, labels):
        """计算特征可分离性"""
        features1 = np.array(features1)
        features2 = np.array(features2)
        labels = np.array(labels).flatten()
        
        # 组合特征 (简单拼接)
        combined_features = np.concatenate([features1, features2], axis=1)
        
        # 正负样本分离
        pos_features = combined_features[labels == 1]
        neg_features = combined_features[labels == 0]
        
        if len(pos_features) < 2 or len(neg_features) < 2:
            return 0.0
        
        # 计算类间距离
        pos_center = np.mean(pos_features, axis=0)
        neg_center = np.mean(neg_features, axis=0)
        inter_class_dist = np.linalg.norm(pos_center - neg_center)
        
        # 计算类内距离
        pos_distances = [np.linalg.norm(f - pos_center) for f in pos_features]
        neg_distances = [np.linalg.norm(f - neg_center) for f in neg_features]
        avg_intra_class_dist = (np.mean(pos_distances) + np.mean(neg_distances)) / 2
        
        # 可分离性比率
        separability = inter_class_dist / (avg_intra_class_dist + 1e-8)
        
        return separability
    
    def visualize_features(self, features1, features2, labels, epoch, save_dir="feature_visualizations"):
        """可视化特征空间"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 组合特征
        combined_features = np.concatenate([features1, features2], axis=1)
        labels = np.array(labels).flatten()
        
        # 使用t-SNE降维
        if len(combined_features) > 50:  # 只有足够的样本才进行可视化
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_features)//4))
            features_2d = tsne.fit_transform(combined_features)
            
            # 绘制
            plt.figure(figsize=(10, 8))
            colors = ['red' if label == 0 else 'blue' for label in labels]
            plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
            plt.title(f'Feature Space Visualization - Epoch {epoch}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend(['Non-matching', 'Matching'])
            plt.grid(True, alpha=0.3)
            
            # 保存
            plt.savefig(f'{save_dir}/features_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def train(self, train_loader, val_loader, epochs=30):
        """训练循环"""
        print(f"🚀 开始训练增强特征提取器")
        
        best_separability = 0.0
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_bce, train_cont, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_results = self.validate_epoch(val_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, separability = val_results[:7]
            val_features1, val_features2, val_labels = val_results[7:]
            
            # 学习率更新
            self.scheduler.step(val_acc)
            
            # 记录
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_bce_loss': train_bce,
                'train_contrastive_loss': train_cont,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'separability': separability,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.history.append(epoch_record)
            
            # 显示结果
            print(f"📈 训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            if self.use_contrastive:
                print(f"   (BCE: {train_bce:.4f}, Contrastive: {train_cont:.4f})")
            print(f"📊 验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            print(f"🎯 特征可分离性: {separability:.4f}")
            print(f"📚 学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 可视化特征空间
            if epoch % 5 == 0:
                self.visualize_features(val_features1, val_features2, val_labels, epoch)
            
            # 保存最佳模型
            if separability > best_separability:
                best_separability = separability
                print(f"  💾 新的最佳可分离性: {best_separability:.4f}")
                torch.save(self.model.state_dict(), 'best_enhanced_model_separability.pth')
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"  💾 新的最佳准确率: {best_acc:.4f}")
                torch.save(self.model.state_dict(), 'best_enhanced_model_accuracy.pth')
            
            # 提前停止检查
            if epoch >= 10 and separability < 0.1:
                print("🔴 特征可分离性过低，提前停止")
                break
        
        print(f"\n🎉 训练完成！")
        print(f"最佳可分离性: {best_separability:.4f}")
        print(f"最佳准确率: {best_acc:.4f}")
        
        return best_separability, best_acc

def main():
    """主函数"""
    print("🔥 增强特征提取器训练")
    print("=" * 50)
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_improved_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=16,  # 减小batch size避免显存不足
        max_points=1000,
        num_workers=2,
        sampling_strategy='ordered'
    )
    
    # 对比实验：普通版本 vs 对比学习版本
    experiments = [
        {"name": "增强特征 (BCE only)", "use_contrastive": False},
        {"name": "增强特征 + 对比学习", "use_contrastive": True, "contrastive_weight": 0.3},
        {"name": "增强特征 + 强对比学习", "use_contrastive": True, "contrastive_weight": 0.7}
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n🔬 实验: {exp['name']}")
        print("=" * 60)
        
        try:
            # 创建训练器
            trainer = EnhancedTrainer(
                use_contrastive=exp["use_contrastive"],
                contrastive_weight=exp.get("contrastive_weight", 0.5)
            )
            
            # 训练
            best_sep, best_acc = trainer.train(train_loader, val_loader, epochs=20)
            
            results[exp["name"]] = {
                "best_separability": best_sep,
                "best_accuracy": best_acc,
                "final_epoch": len(trainer.history),
                "history": trainer.history
            }
            
            print(f"📊 {exp['name']} 结果:")
            print(f"  最佳可分离性: {best_sep:.4f}")
            print(f"  最佳准确率: {best_acc:.4f}")
            
        except Exception as e:
            print(f"❌ {exp['name']} 实验失败: {e}")
            results[exp["name"]] = {"error": str(e)}
    
    # 结果对比
    print(f"\n" + "=" * 70)
    print("📈 实验结果对比")
    print("=" * 70)
    
    best_exp = None
    best_separability = 0.0
    
    for name, result in results.items():
        if "error" not in result:
            sep = result["best_separability"]
            acc = result["best_accuracy"]
            print(f"{name}:")
            print(f"  可分离性: {sep:.4f}")
            print(f"  准确率: {acc:.4f}")
            
            if sep > best_separability:
                best_separability = sep
                best_exp = name
        else:
            print(f"{name}: 失败 ({result['error']})")
    
    if best_exp:
        print(f"\n🏆 最佳方法: {best_exp}")
        print(f"🎯 最佳可分离性: {best_separability:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_feature_results_{timestamp}.json"
    
    # 简化结果用于保存
    simplified_results = {}
    for name, result in results.items():
        if "history" in result:
            simplified_results[name] = {
                "best_separability": result["best_separability"],
                "best_accuracy": result["best_accuracy"],
                "final_epoch": result["final_epoch"]
            }
        else:
            simplified_results[name] = result
    
    with open(results_file, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"📁 实验结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    results = main()