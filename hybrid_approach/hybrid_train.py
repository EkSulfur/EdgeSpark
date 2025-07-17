"""
EdgeSpark 混合方法训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_approach.hybrid_network import HybridEdgeMatchingNet
from simplified_approach.dataset_simple import create_simple_dataloaders

class HybridTrainer:
    """混合方法训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建模型
        self.model = HybridEdgeMatchingNet(**config['model']).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma']
        )
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 历史记录
        self.history = []
        
        # 最佳结果跟踪
        self.best_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(loader):
            batch_start = time.time()
            
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
            
            batch_time = time.time() - batch_start
            
            if batch_idx % 10 == 0:
                print(f'    Batch {batch_idx:3d}/{len(loader):3d} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Time: {batch_time:.2f}s')
        
        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        
        epoch_time = time.time() - epoch_start
        print(f'    Epoch训练用时: {epoch_time:.1f}s')
        
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
    
    def train(self, train_loader, val_loader, epochs):
        """训练循环"""
        print(f"🚀 开始训练混合方法")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"采样次数: {self.config['model']['num_samples']}")
        print(f"采样方法: {self.config['model']['sample_method']}")
        print(f"集成方法: {self.config['model']['ensemble_method']}")
        
        start_time = time.time()
        
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
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                print(f"  💾 新的最佳准确率: {self.best_acc:.4f}")
                
                # 保存模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'epoch': epoch + 1,
                    'best_acc': self.best_acc,
                    'history': self.history
                }, os.path.join(self.save_dir, 'best_hybrid_model.pth'))
            
            # 早停检查
            if epoch - self.best_epoch + 1 >= self.config['training']['early_stopping']:
                print(f"🔴 早停: {self.config['training']['early_stopping']} epochs无提升")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！总用时: {total_time/60:.1f}分钟")
        print(f"🏆 最佳准确率: {self.best_acc:.4f} (epoch {self.best_epoch})")
        
        # 保存训练历史
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.best_acc, self.history

def create_hybrid_configs():
    """创建不同的混合配置进行实验"""
    base_config = {
        'model': {
            'max_points': 1000,
            'num_samples': 5,
            'sample_method': 'diversified',
            'ensemble_method': 'weighted_average'
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'step_size': 10,
            'gamma': 0.5,
            'early_stopping': 8
        },
        'data': {
            'batch_size': 16,  # 减少batch size应对计算开销
            'max_points': 1000,
            'num_workers': 4
        }
    }
    
    # 实验配置
    configs = []
    
    # 实验1: 不同采样次数
    for num_samples in [3, 5, 7]:
        config = base_config.copy()
        config['model'] = base_config['model'].copy()
        config['model']['num_samples'] = num_samples
        config['name'] = f'hybrid_samples_{num_samples}'
        config['save_dir'] = f'hybrid_experiments/samples_{num_samples}_{datetime.now().strftime("%m%d_%H%M")}'
        configs.append(config)
    
    # 实验2: 不同集成方法
    for ensemble_method in ['simple_average', 'weighted_average', 'confidence_weighted']:
        config = base_config.copy()
        config['model'] = base_config['model'].copy()
        config['model']['ensemble_method'] = ensemble_method
        config['name'] = f'hybrid_ensemble_{ensemble_method}'
        config['save_dir'] = f'hybrid_experiments/ensemble_{ensemble_method}_{datetime.now().strftime("%m%d_%H%M")}'
        configs.append(config)
    
    # 实验3: 不同采样方法
    for sample_method in ['diversified', 'random']:
        config = base_config.copy()
        config['model'] = base_config['model'].copy()
        config['model']['sample_method'] = sample_method
        config['name'] = f'hybrid_sample_{sample_method}'
        config['save_dir'] = f'hybrid_experiments/sample_{sample_method}_{datetime.now().strftime("%m%d_%H%M")}'
        configs.append(config)
    
    return configs

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EdgeSpark混合方法训练')
    parser.add_argument('--experiment', type=str, default='all', 
                       help='实验类型: all, samples, ensemble, sample_method')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    args = parser.parse_args()
    
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
    
    # 获取配置
    configs = create_hybrid_configs()
    
    # 根据实验类型选择配置
    if args.experiment == 'samples':
        configs = [c for c in configs if 'samples_' in c['name']]
    elif args.experiment == 'ensemble':
        configs = [c for c in configs if 'ensemble_' in c['name']]
    elif args.experiment == 'sample_method':
        configs = [c for c in configs if 'sample_' in c['name']]
    
    # 快速测试模式
    if args.quick:
        configs = configs[:1]
        args.epochs = 5
        print("🔥 快速测试模式")
    
    # 运行实验
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"🧪 实验 {i+1}/{len(configs)}: {config['name']}")
        print(f"{'='*60}")
        
        # 创建训练器
        trainer = HybridTrainer(config)
        
        # 开始训练
        best_acc, history = trainer.train(train_loader, val_loader, args.epochs)
        
        # 记录结果
        results.append({
            'name': config['name'],
            'config': config,
            'best_acc': best_acc,
            'final_epoch': len(history)
        })
        
        print(f"✅ 实验完成: {config['name']}, 最佳准确率: {best_acc:.4f}")
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 实验总结")
    print(f"{'='*60}")
    
    # 按准确率排序
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['name']}: {result['best_acc']:.4f}")
    
    # 保存总结
    summary_path = f'hybrid_experiments/summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 结果保存在: {summary_path}")
    
    # 显示最佳结果
    if results:
        best_result = results[0]
        print(f"\n🏆 最佳结果:")
        print(f"   实验: {best_result['name']}")
        print(f"   准确率: {best_result['best_acc']:.4f}")
        print(f"   配置: {best_result['config']['model']}")
        
        # 与baseline比较
        baseline_acc = 0.6095  # final_approach的准确率
        improvement = best_result['best_acc'] - baseline_acc
        print(f"\n📈 相对于baseline ({baseline_acc:.4f}):")
        if improvement > 0:
            print(f"   提升: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
        else:
            print(f"   下降: {improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")

if __name__ == "__main__":
    main()