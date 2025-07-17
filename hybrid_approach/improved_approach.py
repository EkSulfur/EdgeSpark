"""
基于实验结果的改进方案
不使用多采样，而是专注于特征工程和网络架构优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class EnhancedEdgeEncoder(nn.Module):
    """
    增强的边缘编码器
    基于实验结果，专注于更好的特征表示而非多采样
    """
    def __init__(self, max_points=1000, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        
        # 1. 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ) for k in [3, 5, 7, 9]  # 多种内核尺寸
        ])
        
        # 2. 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(64 * 4, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 3. 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 4. 全局和局部特征结合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 5. 最终投影
        self.final_proj = nn.Sequential(
            nn.Linear(256 * 2, feature_dim),  # global + max pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, points):
        """
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, feature_dim)
        """
        x = points.transpose(1, 2)  # (batch_size, 2, num_points)
        
        # 1. 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feature = conv(x)
            multi_scale_features.append(feature)
        
        # 2. 特征融合
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # 3. 注意力权重
        attention_weights = self.attention(fused)
        attended = fused * attention_weights
        
        # 4. 全局特征提取
        global_feat = self.global_pool(attended).squeeze(-1)
        max_feat = self.max_pool(attended).squeeze(-1)
        
        # 5. 特征组合
        combined = torch.cat([global_feat, max_feat], dim=1)
        final_features = self.final_proj(combined)
        
        return final_features

class GeometricFeatureExtractor(nn.Module):
    """
    几何特征提取器
    提取边缘的几何特性
    """
    def __init__(self, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 几何特征编码器
        self.geo_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10个几何特征
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
    def extract_geometric_features(self, points):
        """
        提取几何特征
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            geo_features: (batch_size, 10)
        """
        batch_size = points.shape[0]
        features = []
        
        for b in range(batch_size):
            pts = points[b].cpu().numpy()
            
            # 1. 边界框特征
            bbox_min = np.min(pts, axis=0)
            bbox_max = np.max(pts, axis=0)
            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            
            # 2. 统计特征
            centroid = np.mean(pts, axis=0)
            std = np.std(pts, axis=0)
            
            # 3. 形状特征
            # 计算到质心的距离
            distances = np.linalg.norm(pts - centroid, axis=1)
            dist_mean = np.mean(distances)
            dist_std = np.std(distances)
            
            # 组合特征
            geo_feat = np.concatenate([
                bbox_center,    # 2
                bbox_size,      # 2
                centroid,       # 2
                std,           # 2
                [dist_mean],   # 1
                [dist_std]     # 1
            ])
            
            features.append(geo_feat)
        
        features = torch.FloatTensor(features).to(points.device)
        return features
    
    def forward(self, points):
        """
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            geo_features: (batch_size, feature_dim)
        """
        geo_features = self.extract_geometric_features(points)
        encoded = self.geo_encoder(geo_features)
        return encoded

class ImprovedEdgeMatchingNet(nn.Module):
    """
    改进的边缘匹配网络
    基于实验结果，专注于更好的特征表示
    """
    def __init__(self, max_points=1000, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        
        # 增强的边缘编码器
        self.edge_encoder = EnhancedEdgeEncoder(max_points, feature_dim)
        
        # 几何特征提取器
        self.geo_extractor = GeometricFeatureExtractor(feature_dim // 4)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            # 多种特征组合
            nn.Linear(feature_dim * 2 + feature_dim + feature_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, points1, points2):
        """
        Args:
            points1: (batch_size, num_points1, 2)
            points2: (batch_size, num_points2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        # 1. 边缘特征编码
        edge_feat1 = self.edge_encoder(points1)
        edge_feat2 = self.edge_encoder(points2)
        
        # 2. 几何特征提取
        geo_feat1 = self.geo_extractor(points1)
        geo_feat2 = self.geo_extractor(points2)
        
        # 3. 特征融合
        fused_feat1 = self.feature_fusion(torch.cat([edge_feat1, geo_feat1], dim=1))
        fused_feat2 = self.feature_fusion(torch.cat([edge_feat2, geo_feat2], dim=1))
        
        # 4. 多种相似度计算
        # 基本组合
        diff = fused_feat1 - fused_feat2
        hadamard = fused_feat1 * fused_feat2
        
        # 相似度度量
        cosine_sim = F.cosine_similarity(fused_feat1, fused_feat2, dim=1, eps=1e-8)
        euclidean_dist = torch.norm(diff, p=2, dim=1, keepdim=True)
        
        # 5. 特征组合
        combined = torch.cat([
            fused_feat1,           # 第一个特征
            fused_feat2,           # 第二个特征
            diff,                  # 差值
            hadamard,              # 哈达玛积
            cosine_sim.unsqueeze(1)  # 余弦相似度
        ], dim=1)
        
        # 6. 匹配预测
        match_logits = self.matching_net(combined)
        
        return match_logits

# 测试函数
def test_improved_network():
    """测试改进网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建网络
    model = ImprovedEdgeMatchingNet(max_points=1000, feature_dim=128).to(device)
    
    # 测试数据
    batch_size = 8
    points1 = torch.randn(batch_size, 800, 2).to(device)
    points2 = torch.randn(batch_size, 900, 2).to(device)
    
    # 前向传播
    print("前向传播测试...")
    with torch.no_grad():
        output = model(points1, points2)
    
    print(f"网络输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 梯度测试
    print("\n梯度测试...")
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
    test_improved_network()