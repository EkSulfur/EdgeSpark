import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SimpleSegmentEncoder(nn.Module):
    """
    简化的边缘段编码器
    使用简单的1D CNN + 池化
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简单的1D CNN
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=5, padding=2)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, points):
        """
        编码点云序列
        Args:
            points: (batch_size, sequence_length, 2)
        Returns:
            features: (batch_size, output_dim)
        """
        # 转换维度 (batch_size, sequence_length, 2) -> (batch_size, 2, sequence_length)
        x = points.transpose(1, 2)
        
        # 1D CNN编码
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 全局平均池化
        x = self.global_pool(x).squeeze(-1)  # (batch_size, output_dim)
        
        return x

class SimpleSegmentSampler(nn.Module):
    """
    简化的段采样器
    使用确定性采样而非随机采样
    """
    def __init__(self, segment_length=64, num_segments=8):
        super().__init__()
        self.segment_length = segment_length
        self.num_segments = num_segments
        
    def forward(self, points):
        """
        从点云中采样段
        Args:
            points: (batch_size, N, 2)
        Returns:
            segments: (batch_size, num_segments, segment_length, 2)
        """
        batch_size, N, _ = points.shape
        
        if N < self.segment_length:
            # 如果点数不够，重复采样
            indices = torch.randint(0, N, (batch_size, self.num_segments, self.segment_length), 
                                  device=points.device)
            segments = torch.gather(points.unsqueeze(1).expand(-1, self.num_segments, -1, -1), 
                                  2, indices.unsqueeze(-1).expand(-1, -1, -1, 2))
        else:
            # 均匀采样
            segments = []
            for i in range(self.num_segments):
                start_idx = i * (N - self.segment_length) // max(1, self.num_segments - 1)
                if start_idx + self.segment_length > N:
                    start_idx = N - self.segment_length
                segment = points[:, start_idx:start_idx + self.segment_length]
                segments.append(segment)
            segments = torch.stack(segments, dim=1)
        
        return segments

class SimpleSimilarityMatcher(nn.Module):
    """
    简化的相似度匹配器
    使用简单的余弦相似度和注意力机制
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征变换 - 简化
        self.query_proj = nn.Linear(feature_dim, feature_dim // 2)
        self.key_proj = nn.Linear(feature_dim, feature_dim // 2)
        self.value_proj = nn.Linear(feature_dim, feature_dim // 2)
        
        # 注意力权重
        self.attention_weights = nn.Linear(feature_dim, 1)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, features1, features2):
        """
        计算两组特征的匹配分数
        Args:
            features1: (batch_size, num_segments1, feature_dim)
            features2: (batch_size, num_segments2, feature_dim)
        Returns:
            match_score: (batch_size, 1)
        """
        batch_size = features1.shape[0]
        
        # 特征变换
        q = self.query_proj(features1)  # (batch_size, num_segments1, feature_dim)
        k = self.key_proj(features2)    # (batch_size, num_segments2, feature_dim)
        v = self.value_proj(features2)  # (batch_size, num_segments2, feature_dim)
        
        # 计算注意力权重 (简化的交叉注意力)
        # q @ k.T -> (batch_size, num_segments1, num_segments2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.feature_dim // 2)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权聚合
        # (batch_size, num_segments1, num_segments2) @ (batch_size, num_segments2, feature_dim//2)
        # -> (batch_size, num_segments1, feature_dim//2)
        attended_features = torch.matmul(attention_weights, v)
        
        # 特征融合 (拼接而非相加)
        fused_features = torch.cat([q, attended_features], dim=-1)  # (batch_size, num_segments1, feature_dim)
        
        # 全局池化
        pooled_features = fused_features.mean(dim=1)  # (batch_size, feature_dim)
        
        # 分类
        match_score = self.classifier(pooled_features)
        
        return match_score

class SimpleEdgeSparkNet(nn.Module):
    """
    简化版EdgeSpark网络
    移除复杂的多次采样和马氏距离计算
    """
    def __init__(self, 
                 segment_length=64,
                 num_segments=8,
                 feature_dim=128,
                 hidden_dim=128):
        super().__init__()
        
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.feature_dim = feature_dim
        
        # 组件
        self.sampler = SimpleSegmentSampler(segment_length, num_segments)
        self.encoder = SimpleSegmentEncoder(input_dim=2, hidden_dim=hidden_dim, output_dim=feature_dim)
        self.matcher = SimpleSimilarityMatcher(feature_dim)
        
    def forward(self, points1, points2):
        """
        前向传播
        Args:
            points1: (batch_size, N1, 2)
            points2: (batch_size, N2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        batch_size = points1.shape[0]
        
        # 1. 段采样
        segments1 = self.sampler(points1)  # (batch_size, num_segments, segment_length, 2)
        segments2 = self.sampler(points2)  # (batch_size, num_segments, segment_length, 2)
        
        # 2. 特征编码
        # 重塑为 (batch_size * num_segments, segment_length, 2)
        segments1_flat = segments1.view(-1, self.segment_length, 2)
        segments2_flat = segments2.view(-1, self.segment_length, 2)
        
        # 编码
        features1_flat = self.encoder(segments1_flat)  # (batch_size * num_segments, feature_dim)
        features2_flat = self.encoder(segments2_flat)  # (batch_size * num_segments, feature_dim)
        
        # 重塑回 (batch_size, num_segments, feature_dim)
        features1 = features1_flat.view(batch_size, self.num_segments, self.feature_dim)
        features2 = features2_flat.view(batch_size, self.num_segments, self.feature_dim)
        
        # 3. 相似度匹配
        match_logits = self.matcher(features1, features2)
        
        return match_logits

# 测试函数
def test_simple_network():
    """测试简化网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建网络
    net = SimpleEdgeSparkNet(
        segment_length=64,
        num_segments=8,
        feature_dim=128,
        hidden_dim=128
    ).to(device)
    
    # 测试数据
    batch_size = 4
    points1 = torch.randn(batch_size, 800, 2).to(device)
    points2 = torch.randn(batch_size, 900, 2).to(device)
    
    # 前向传播
    print("前向传播测试...")
    with torch.no_grad():
        output = net(points1, points2)
    
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output.squeeze()}")
    
    # 参数统计
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 梯度测试
    print("\n梯度测试...")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 虚拟标签
    labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    # 训练步骤
    optimizer.zero_grad()
    output = net(points1, points2)
    loss = F.binary_cross_entropy_with_logits(output, labels)
    loss.backward()
    
    # 检查梯度
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
    print(f"梯度范数: {grad_norm:.4f}")
    print(f"损失: {loss.item():.4f}")
    
    optimizer.step()
    print("梯度更新成功!")

if __name__ == "__main__":
    test_simple_network()