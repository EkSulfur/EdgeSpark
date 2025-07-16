import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from network_improved import EdgeSparkNet
from dataset_loader import create_dataloaders

class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建TensorBoard日志
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))
        
        # 初始化网络
        self.model = EdgeSparkNet(**config['model']).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=config['training']['patience'],
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移到设备
            source_points = batch['source_points'].to(self.device)
            target_points = batch['target_points'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(source_points, target_points)
            
            # 计算损失
            loss = self.criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            all_predictions.extend((predictions > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # 计算平均损失和准确率
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
                # 数据移到设备
                source_points = batch['source_points'].to(self.device)
                target_points = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                predictions = self.model(source_points, target_points)
                
                # 计算损失
                loss = self.criterion(predictions, labels)
                
                # 统计
                total_loss += loss.item()
                all_predictions.extend((predictions > 0.5).cpu().numpy())
                all_probabilities.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        # 计算AUC
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_history = checkpoint['train_history']
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, start_epoch=0):
        """主训练循环"""
        print("开始训练...")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['val_auc'].append(val_auc)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('F1/Val', val_f1, epoch)
            self.writer.add_scalar('AUC/Val', val_auc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                is_best = True
            
            # 保存检查点
            self.save_checkpoint(epoch + 1, is_best)
            
            # 早停检查
            if epoch - self.best_epoch + 1 >= self.config['training']['early_stopping']:
                print(f"早停：{self.config['training']['early_stopping']} epochs没有改善")
                break
        
        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        self.writer.close()

def create_config():
    """创建训练配置"""
    config = {
        'model': {
            'segment_length': 32,
            'n1': 16,
            'n2': 16,
            'feature_dim': 256,
            'hidden_channels': 64,
            'temperature': 1.0,
            'num_samples': 3
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 10,
            'early_stopping': 20
        },
        'data': {
            'batch_size': 16,
            'max_points': 1500,
            'num_workers': 4
        },
        'save_dir': f'experiments/exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    return config

def main():
    """主函数"""
    # 创建配置
    config = create_config()
    
    # 保存配置
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=config['data']['batch_size'],
        max_points=config['data']['max_points'],
        num_workers=config['data']['num_workers']
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    print(f"实验结果保存在: {config['save_dir']}")

if __name__ == "__main__":
    main()