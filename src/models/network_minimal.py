import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MinimalEdgeEncoder(nn.Module):
    """
    极简的边缘编码器
    直接对整个边缘点云进行编码
    """
    def __init__(self, max_points=1000, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        
        # 点云编码 - 简单的全连接层
        self.point_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # 全局特征聚合
        self.global_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, points):
        """
        编码点云
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, feature_dim)
        """
        batch_size, num_points, _ = points.shape
        
        # 点级别特征提取
        point_features = self.point_encoder(points)  # (batch_size, num_points, feature_dim)
        
        # 全局池化
        global_features = torch.mean(point_features, dim=1)  # (batch_size, feature_dim)
        
        # 全局特征编码
        output_features = self.global_encoder(global_features)
        
        return output_features

class MinimalEdgeSparkNet(nn.Module):
    """
    极简版EdgeSpark网络
    直接比较两个边缘点云的全局特征
    """
    def __init__(self, max_points=1000, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        
        # 边缘编码器
        self.edge_encoder = MinimalEdgeEncoder(max_points, feature_dim)
        
        # 特征融合和分类
        self.classifier = nn.Sequential(
            # 拼接两个特征
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1)
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
        # 编码两个边缘点云
        features1 = self.edge_encoder(points1)  # (batch_size, feature_dim)
        features2 = self.edge_encoder(points2)  # (batch_size, feature_dim)
        
        # 特征融合
        combined_features = torch.cat([features1, features2], dim=1)  # (batch_size, feature_dim * 2)
        
        # 分类
        match_logits = self.classifier(combined_features)
        
        return match_logits

# 测试函数
def test_minimal_network():
    """测试极简网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建网络
    model = MinimalEdgeSparkNet(max_points=1000, feature_dim=128).to(device)
    
    # 测试数据
    batch_size = 8
    points1 = torch.randn(batch_size, 800, 2).to(device)
    points2 = torch.randn(batch_size, 900, 2).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(points1, points2)
    
    print(f"网络输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 梯度测试
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    optimizer.zero_grad()
    output = model(points1, points2)
    loss = F.binary_cross_entropy_with_logits(output, labels)
    loss.backward()
    optimizer.step()
    
    print(f"损失: {loss.item():.4f}")
    print("测试成功!")

if __name__ == "__main__":
    test_minimal_network()