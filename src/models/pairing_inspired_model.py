"""
基于PairingNet的EdgeSpark模型改进版
充分利用邻接矩阵和空间特征进行碎片匹配
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple

class GraphConvLayer(nn.Module):
    """
    图卷积层 - 基于PairingNet的GCN设计
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 线性变换
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 节点特征 (batch_size, num_nodes, in_features)
            adj: 邻接矩阵 (batch_size, num_nodes, num_nodes)
        Returns:
            output: 输出特征 (batch_size, num_nodes, out_features)
        """
        # 线性变换
        x = self.linear(x)  # (batch_size, num_nodes, out_features)
        
        # 图卷积：邻接矩阵与特征的矩阵乘法
        x = torch.bmm(adj, x)  # (batch_size, num_nodes, out_features)
        
        # 批归一化（需要调整维度）
        batch_size, num_nodes, features = x.shape
        x = x.view(-1, features)  # (batch_size * num_nodes, features)
        x = self.batch_norm(x)
        x = x.view(batch_size, num_nodes, features)
        
        # 激活和dropout
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class MultiScaleGCN(nn.Module):
    """
    多尺度图卷积网络
    """
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 构建多层GCN
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(GraphConvLayer(dims[i], dims[i+1], dropout))
        
        # 跳跃连接的投影层
        self.skip_projections = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.skip_projections.append(nn.Linear(input_dim, hidden_dim))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, num_nodes, input_dim)
            adj: 邻接矩阵 (batch_size, num_nodes, num_nodes)
        Returns:
            output: 输出特征 (batch_size, num_nodes, hidden_dims[-1])
        """
        original_x = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            
            # 跳跃连接
            if i < len(self.skip_projections):
                skip = self.skip_projections[i](original_x)
                x = x + skip
        
        return x

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    """
    def __init__(self, feature_dim: int, spatial_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        
        # 注意力计算
        self.attention = nn.Sequential(
            nn.Linear(feature_dim + spatial_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            features: 特征 (batch_size, num_nodes, feature_dim)
            spatial: 空间特征 (batch_size, num_nodes, spatial_dim)
        Returns:
            attended_features: 注意力加权后的特征 (batch_size, num_nodes, feature_dim)
        """
        # 拼接特征和空间信息
        combined = torch.cat([features, spatial], dim=-1)
        
        # 计算注意力权重
        attention_weights = self.attention(combined)  # (batch_size, num_nodes, 1)
        
        # 应用注意力权重
        attended_features = features * attention_weights
        
        return attended_features

class PairingInspiredEncoder(nn.Module):
    """
    基于PairingNet的编码器
    """
    def __init__(self, max_points: int = 1000, use_spatial: bool = True):
        super().__init__()
        self.max_points = max_points
        self.use_spatial = use_spatial
        
        # 输入维度：2D坐标 + 可选的空间特征
        input_dim = 2
        if use_spatial:
            input_dim += 4  # 角度、曲率、距离1、距离2
        
        # 多尺度GCN
        self.gcn = MultiScaleGCN(
            input_dim=input_dim,
            hidden_dims=[64, 128, 256, 128],
            dropout=0.1
        )
        
        # 空间注意力
        if use_spatial:
            self.spatial_attention = SpatialAttention(128, 4)
        
        # 全局池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 最终投影
        self.final_projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
    def forward(self, points: torch.Tensor, adj: torch.Tensor, 
                spatial: Optional[torch.Tensor] = None, 
                length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            points: 点云坐标 (batch_size, num_points, 2)
            adj: 邻接矩阵 (batch_size, num_points, num_points)
            spatial: 空间特征 (batch_size, num_points, 4)
            length: 实际点云长度 (batch_size, 1)
        Returns:
            features: 全局特征 (batch_size, 64)
        """
        # 准备输入特征
        if self.use_spatial and spatial is not None:
            x = torch.cat([points, spatial], dim=-1)
        else:
            x = points
        
        # 多尺度GCN
        x = self.gcn(x, adj)  # (batch_size, num_points, 128)
        
        # 空间注意力
        if self.use_spatial and spatial is not None:
            x = self.spatial_attention(x, spatial)
        
        # 处理变长序列（使用mask）
        if length is not None:
            # 创建mask
            batch_size, max_len = x.shape[0], x.shape[1]
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len)
            mask = mask < length  # (batch_size, max_len)
            
            # 应用mask
            x = x * mask.unsqueeze(-1).float()
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch_size, 128, num_points)
        x = self.global_pool(x)  # (batch_size, 128)
        
        # 最终投影
        x = self.final_projection(x)  # (batch_size, 64)
        
        return x

class FocalLoss(nn.Module):
    """
    FocalLoss实现
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            inputs: 预测logits (batch_size, 1)
            targets: 真实标签 (batch_size, 1)
        Returns:
            loss: focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

class PairingInspiredMatchingNet(nn.Module):
    """
    基于PairingNet的匹配网络
    """
    def __init__(self, max_points: int = 1000, use_spatial: bool = True, temperature: float = 1.0):
        super().__init__()
        self.max_points = max_points
        self.use_spatial = use_spatial
        self.temperature = temperature
        
        # 编码器
        self.encoder = PairingInspiredEncoder(max_points, use_spatial)
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            nn.Linear(64 * 3 + 1, 128),  # 64*2 + 64 + 1 (拼接+差值+相似度)
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
        
    def compute_temperature_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        计算温度缩放的相似度
        Args:
            feat1: 特征1 (batch_size, 64)
            feat2: 特征2 (batch_size, 64)
        Returns:
            similarity: 温度缩放后的相似度 (batch_size, 1)
        """
        # L2归一化
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(feat1_norm * feat2_norm, dim=1, keepdim=True)
        
        # 温度缩放
        scaled_sim = cos_sim / self.temperature
        
        return scaled_sim
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            batch: 批次数据
        Returns:
            logits: 匹配预测 (batch_size, 1)
        """
        # 提取输入
        source_points = batch['source_points']
        target_points = batch['target_points']
        source_adj = batch['source_adj']
        target_adj = batch['target_adj']
        source_length = batch['source_length']
        target_length = batch['target_length']
        
        source_spatial = batch.get('source_spatial', None)
        target_spatial = batch.get('target_spatial', None)
        
        # 编码
        source_feat = self.encoder(source_points, source_adj, source_spatial, source_length)
        target_feat = self.encoder(target_points, target_adj, target_spatial, target_length)
        
        # 特征组合
        diff_feat = source_feat - target_feat  # 差值特征
        temp_sim = self.compute_temperature_similarity(source_feat, target_feat)  # 温度相似度
        
        # 拼接所有特征
        combined_feat = torch.cat([source_feat, target_feat, diff_feat, temp_sim], dim=1)
        
        # 匹配预测
        logits = self.matching_net(combined_feat)
        
        return logits

class PairingInspiredTrainer:
    """
    基于PairingNet的训练器
    """
    def __init__(self, max_points: int = 1000, use_spatial: bool = True, 
                 temperature: float = 1.0, alpha: float = 0.55, gamma: float = 2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = PairingInspiredMatchingNet(
            max_points=max_points,
            use_spatial=use_spatial,
            temperature=temperature
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        
        # 损失函数
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        
        # 历史记录
        self.history = []
        
        print(f"🚀 PairingNet风格训练器初始化:")
        print(f"   - 最大点数: {max_points}")
        print(f"   - 使用空间特征: {use_spatial}")
        print(f"   - 温度参数: {temperature}")
        print(f"   - FocalLoss: α={alpha}, γ={gamma}")
        print(f"   - 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(loader):
            # 移动到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            labels = batch['label']
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f'    Batch {batch_idx:3d}/{len(loader):3d} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f}')
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                labels = batch['label']
                
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 30):
        """训练主循环"""
        print(f"🎯 开始训练 PairingNet风格模型...")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 学习率更新
            self.scheduler.step()
            
            # 记录历史
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 显示结果
            print(f"📈 训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"📊 验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"📚 学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"  💾 新的最佳准确率: {best_acc:.4f}")
                torch.save(self.model.state_dict(), 'best_pairing_inspired_model.pth')
            
            # 提前停止
            if epoch >= 10 and val_acc < 0.55:
                print("🔴 验证准确率过低，提前停止")
                break
        
        print(f"\n🎉 训练完成！最佳准确率: {best_acc:.4f}")
        return best_acc

# 测试函数
def test_pairing_inspired_model():
    """测试模型"""
    print("🔍 测试PairingNet风格模型...")
    
    try:
        # 创建数据加载器
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
        from pairing_inspired_dataloader import create_pairing_inspired_dataloaders
        
        train_loader, val_loader, test_loader = create_pairing_inspired_dataloaders(
            "dataset/train_set.pkl",
            "dataset/valid_set.pkl",
            "dataset/test_set.pkl",
            batch_size=8,
            max_points=1000,
            num_workers=0,
            use_spatial_features=True
        )
        
        # 创建训练器
        trainer = PairingInspiredTrainer(
            max_points=1000,
            use_spatial=True,
            temperature=1.0,
            alpha=0.55,
            gamma=2.0
        )
        
        # 测试一个前向传播
        for batch in train_loader:
            # 移动到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(trainer.device)
            
            logits = trainer.model(batch)
            print(f"✅ 模型输出形状: {logits.shape}")
            print(f"   - 样本标签: {batch['label'].squeeze()}")
            print(f"   - 预测概率: {torch.sigmoid(logits).squeeze()}")
            break
        
        print("✅ 模型测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pairing_inspired_model()