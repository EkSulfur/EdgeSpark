import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from network_minimal import MinimalEdgeSparkNet
from dataset_simple import create_simple_dataloaders

class MinimalTrainer:
    """极简版训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建TensorBoard
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))
        
        # 初始化网络
        self.model = MinimalEdgeSparkNet(**config['model']).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(  # 使用Adam
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': [],
            'lr': []
        }
        
        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据准备
            source_points = batch['source_points'].to(self.device)
            target_points = batch['target_points'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(source_points, target_points)
            
            # 计算损失
            loss = self.criterion(predictions, labels)
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: 跳过无效损失 {loss.item()}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 参数更新
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 进度显示
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Batch {batch_idx:3d}/{len(train_loader):3d} | '
                      f'Loss: {loss.item():.4f} | LR: {current_lr:.6f}')
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 数据准备
                source_points = batch['source_points'].to(self.device)
                target_points = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                predictions = self.model(source_points, target_points)
                
                # 计算损失
                loss = self.criterion(predictions, labels)
                
                # 统计
                total_loss += loss.item()
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).float()
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary', zero_division=0
            )
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  💾 保存最佳模型: acc={self.best_val_acc:.4f}")
    
    def train(self, train_loader, val_loader):
        """主训练循环"""
        print(f"🚀 开始训练极简版EdgeSpark")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()
            print(f"\n📊 Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['val_auc'].append(val_auc)
            self.train_history['lr'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Acc/Train', train_acc, epoch)
            self.writer.add_scalar('Acc/Val', val_acc, epoch)
            self.writer.add_scalar('F1/Val', val_f1, epoch)
            self.writer.add_scalar('AUC/Val', val_auc, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 显示结果
            epoch_time = time.time() - epoch_start
            print(f"📈 训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"📊 验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            print(f"⏱️  用时: {epoch_time:.1f}s, LR: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
            
            # 保存检查点
            self.save_checkpoint(epoch + 1, is_best)
            
            # 早停检查
            if epoch - self.best_epoch + 1 >= self.config['training']['early_stopping']:
                print(f"🔴 早停: {self.config['training']['early_stopping']} epochs无提升")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成! 总用时: {total_time/60:.1f}分钟")
        print(f"🏆 最佳结果: Acc={self.best_val_acc:.4f} (epoch {self.best_epoch})")
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        self.writer.close()

def create_minimal_config():
    """创建极简训练配置"""
    config = {
        'model': {
            'max_points': 1000,
            'feature_dim': 128
        },
        'training': {
            'epochs': 30,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping': 10
        },
        'data': {
            'batch_size': 64,  # 增加batch size
            'max_points': 1000,
            'num_workers': 4
        },
        'save_dir': f'experiments/minimal_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EdgeSpark极简版训练')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    args = parser.parse_args()
    
    # 创建配置
    config = create_minimal_config()
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['data']['batch_size'] = args.batch_size
    
    # 保存配置
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建数据加载器
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=config['data']['batch_size'],
        max_points=config['data']['max_points'],
        num_workers=config['data']['num_workers']
    )
    
    # 创建训练器
    trainer = MinimalTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    print(f"📁 结果保存在: {config['save_dir']}")

if __name__ == "__main__":
    main()