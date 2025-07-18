import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SegmentSampler(nn.Module):
    """
    边缘段采样器
    从边缘点云中采样固定长度的段
    """
    def __init__(self, segment_length=32):
        super().__init__()
        self.segment_length = segment_length
    
    def forward(self, edge_points, num_samples):
        """
        从边缘点云中采样段
        Args:
            edge_points: (N, 2) 边缘点云
            num_samples: 采样段的数量
        Returns:
            segments: (num_samples, segment_length, 2) 采样的段
        """
        N = edge_points.shape[0]
        segments = []
        
        for _ in range(num_samples):
            # 随机选择起始点，确保不越界
            start_idx = torch.randint(0, max(1, N - self.segment_length + 1), (1,)).item()
            if start_idx + self.segment_length > N:
                # 如果越界，从末尾开始取
                segment = edge_points[N - self.segment_length:N]
            else:
                segment = edge_points[start_idx:start_idx + self.segment_length]
            segments.append(segment)
        
        return torch.stack(segments)

class SegmentEncoder(nn.Module):
    """
    边缘段编码器
    将边缘段编码为固定维度的特征向量
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1D CNN for sequence encoding
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, segments):
        """
        编码边缘段
        Args:
            segments: (batch_size, num_segments, segment_length, 2)
        Returns:
            features: (batch_size, num_segments, output_dim)
        """
        batch_size, num_segments, segment_length, _ = segments.shape
        
        # Reshape for processing
        segments = segments.view(batch_size * num_segments, segment_length, self.input_dim)
        segments = segments.transpose(1, 2)  # (batch_size * num_segments, 2, segment_length)
        
        # CNN encoding
        x = F.relu(self.bn1(self.conv1(segments)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size * num_segments, hidden_dim)
        
        # Projection
        x = self.projection(x)  # (batch_size * num_segments, output_dim)
        
        # Reshape back
        features = x.view(batch_size, num_segments, self.output_dim)
        
        return features

class LearnableMetricSimilarity(nn.Module):
    """
    可学习的度量矩阵相似度计算器
    """
    def __init__(self, feature_dim=256, temperature=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # 可学习的度量矩阵（对称正定）
        self.metric_matrix = nn.Parameter(torch.eye(feature_dim))
        
        # 可选的特征变换
        self.feature_transform = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, features1, features2):
        """
        计算可学习度量下的相似度矩阵
        Args:
            features1: (batch_size, n1, feature_dim)
            features2: (batch_size, n2, feature_dim)
        Returns:
            similarity_matrix: (batch_size, n1, n2)
        """
        # 可选的特征变换
        features1 = self.feature_transform(features1)
        features2 = self.feature_transform(features2)
        
        # 计算特征差异
        features1_expanded = features1.unsqueeze(2)  # (batch_size, n1, 1, feature_dim)
        features2_expanded = features2.unsqueeze(1)  # (batch_size, 1, n2, feature_dim)
        
        diff = features1_expanded - features2_expanded  # (batch_size, n1, n2, feature_dim)
        
        # 使用可学习的度量矩阵计算距离
        # 确保度量矩阵是对称的
        metric_matrix = (self.metric_matrix + self.metric_matrix.t()) / 2
        
        # 计算马氏距离的平方
        # diff @ metric_matrix @ diff.transpose(-2, -1)
        weighted_diff = torch.matmul(diff, metric_matrix)  # (batch_size, n1, n2, feature_dim)
        mahalanobis_dist_sq = torch.sum(weighted_diff * diff, dim=-1)  # (batch_size, n1, n2)
        
        # 转换为相似度（距离越小，相似度越高）
        # 添加数值稳定性：限制指数的输入范围
        stable_input = torch.clamp(-mahalanobis_dist_sq / self.temperature, min=-10, max=10)
        similarity_matrix = torch.exp(stable_input)
        
        return similarity_matrix

class SimilarityMatrixProcessor(nn.Module):
    """
    相似度矩阵处理器
    使用2D CNN处理相似度矩阵，保持空间结构
    """
    def __init__(self, input_channels=1, hidden_channels=64):
        super().__init__()
        
        # 2D CNN layers
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, similarity_matrix):
        """
        处理相似度矩阵
        Args:
            similarity_matrix: (batch_size, n1, n2)
        Returns:
            processed_features: (batch_size, hidden_channels)
        """
        # 添加通道维度
        x = similarity_matrix.unsqueeze(1)  # (batch_size, 1, n1, n2)
        
        # 2D CNN处理
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 全局池化
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch_size, hidden_channels)
        
        return x

class AttentionPooling(nn.Module):
    """
    改进的注意力池化
    从相似度矩阵中找到关键的匹配区域，支持多头注意力
    """
    def __init__(self, feature_dim=64, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        # 确保num_heads能被feature_dim整除
        self.num_heads = min(num_heads, feature_dim)
        if feature_dim % self.num_heads != 0:
            self.num_heads = 8 if feature_dim >= 8 else feature_dim
        self.head_dim = feature_dim // self.num_heads
        
        # 多头注意力
        self.query_proj = nn.Linear(1, feature_dim)
        self.key_proj = nn.Linear(1, feature_dim)
        self.value_proj = nn.Linear(1, feature_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 2560, feature_dim))  # 支持最大50x50的相似度矩阵
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, similarity_matrix):
        """
        多头注意力池化
        Args:
            similarity_matrix: (batch_size, n1, n2)
        Returns:
            pooled_features: (batch_size, feature_dim)
        """
        batch_size, n1, n2 = similarity_matrix.shape
        seq_len = n1 * n2
        
        # 展平相似度矩阵并添加位置编码
        flat_similarity = similarity_matrix.view(batch_size, seq_len, 1)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        
        # 计算查询、键、值
        queries = self.query_proj(flat_similarity) + pos_enc  # (batch_size, seq_len, feature_dim)
        keys = self.key_proj(flat_similarity) + pos_enc
        values = self.value_proj(flat_similarity)
        
        # 多头注意力
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 计算注意力得分
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.view(batch_size, seq_len, self.feature_dim)
        
        # 全局平均池化
        pooled_features = torch.mean(attended_values, dim=1)  # (batch_size, feature_dim)
        
        # 输出投影和规范化
        pooled_features = self.output_proj(pooled_features)
        pooled_features = self.norm(pooled_features)
        
        return pooled_features

class CrossAttention(nn.Module):
    """
    交叉注意力模块
    让两个片段的特征互相关注
    """
    def __init__(self, feature_dim=256, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = min(num_heads, feature_dim)
        if feature_dim % self.num_heads != 0:
            self.num_heads = 8 if feature_dim >= 8 else feature_dim
        self.head_dim = feature_dim // self.num_heads
        
        # 查询、键、值投影
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, features1, features2):
        """
        交叉注意力计算
        Args:
            features1: (batch_size, n1, feature_dim)
            features2: (batch_size, n2, feature_dim)
        Returns:
            enhanced_features1: (batch_size, n1, feature_dim)
            enhanced_features2: (batch_size, n2, feature_dim)
        """
        batch_size, n1, _ = features1.shape
        _, n2, _ = features2.shape
        
        # 计算features1对features2的注意力
        q1 = self.query_proj(features1).view(batch_size, n1, self.num_heads, self.head_dim)
        k2 = self.key_proj(features2).view(batch_size, n2, self.num_heads, self.head_dim)
        v2 = self.value_proj(features2).view(batch_size, n2, self.num_heads, self.head_dim)
        
        # 注意力计算
        attention_scores = torch.matmul(q1, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        attended_values = torch.matmul(attention_weights, v2)
        attended_values = attended_values.view(batch_size, n1, self.feature_dim)
        
        # 残差连接和层归一化
        enhanced_features1 = self.norm1(features1 + self.output_proj(attended_values))
        enhanced_features1 = self.norm2(enhanced_features1 + self.ffn(enhanced_features1))
        
        # 计算features2对features1的注意力
        q2 = self.query_proj(features2).view(batch_size, n2, self.num_heads, self.head_dim)
        k1 = self.key_proj(features1).view(batch_size, n1, self.num_heads, self.head_dim)
        v1 = self.value_proj(features1).view(batch_size, n1, self.num_heads, self.head_dim)
        
        attention_scores = torch.matmul(q2, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_weights, v1)
        attended_values = attended_values.view(batch_size, n2, self.feature_dim)
        
        enhanced_features2 = self.norm1(features2 + self.output_proj(attended_values))
        enhanced_features2 = self.norm2(enhanced_features2 + self.ffn(enhanced_features2))
        
        return enhanced_features1, enhanced_features2

class EdgeSparkNet(nn.Module):
    """
    EdgeSpark改进版网络
    基于边缘信息的碎片匹配网络
    """
    def __init__(self, 
                 segment_length=32,
                 n1=1600,  # 第一个碎片的采样段数
                 n2=1600,  # 第二个碎片的采样段数
                 feature_dim=256,
                 hidden_channels=64,
                 temperature=1.0,
                 num_samples=5):  # 多次采样取平均
        super().__init__()
        
        self.segment_length = segment_length
        self.n1 = n1
        self.n2 = n2
        self.num_samples = num_samples
        
        # 各个模块
        self.segment_sampler = SegmentSampler(segment_length)
        self.segment_encoder = SegmentEncoder(output_dim=feature_dim)
        self.cross_attention = CrossAttention(feature_dim=feature_dim)
        self.similarity_computer = LearnableMetricSimilarity(feature_dim, temperature)
        self.similarity_processor = SimilarityMatrixProcessor(
            input_channels=1, 
            hidden_channels=hidden_channels
        )
        self.attention_pooling = AttentionPooling(feature_dim=hidden_channels)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, edge_points1, edge_points2):
        """
        前向传播
        Args:
            edge_points1: (batch_size, N1, 2) 第一个碎片的边缘点云
            edge_points2: (batch_size, N2, 2) 第二个碎片的边缘点云
        Returns:
            match_logits: (batch_size, 1) 匹配对数几率
        """
        batch_size = edge_points1.shape[0]
        
        # 原本：多次采样取平均，这里单次采样，不取平均
        match_probs = []
        
        for _ in range(1):  # 原本为 self.num_samples
            # 1. 段采样
            segments1_list = []
            segments2_list = []
            
            for i in range(batch_size):
                seg1 = self.segment_sampler(edge_points1[i], self.n1)
                seg2 = self.segment_sampler(edge_points2[i], self.n2)
                segments1_list.append(seg1)
                segments2_list.append(seg2)
            
            segments1 = torch.stack(segments1_list)  # (batch_size, n1, segment_length, 2)
            segments2 = torch.stack(segments2_list)  # (batch_size, n2, segment_length, 2)
            
            # 2. 特征编码
            features1 = self.segment_encoder(segments1)  # (batch_size, n1, feature_dim)
            features2 = self.segment_encoder(segments2)  # (batch_size, n2, feature_dim)
            
            # 3. 交叉注意力增强特征
            enhanced_features1, enhanced_features2 = self.cross_attention(features1, features2)
            
            # 4. 相似度计算（使用增强后的特征）
            similarity_matrix = self.similarity_computer(enhanced_features1, enhanced_features2)  # (batch_size, n1, n2)
            
            # 5. 相似度矩阵处理
            processed_features = self.similarity_processor(similarity_matrix)  # (batch_size, hidden_channels)
            
            # 6. 注意力池化（增强特征）
            attention_features = self.attention_pooling(similarity_matrix)  # (batch_size, hidden_channels)
            
            # 7. 特征融合
            fused_features = processed_features + attention_features  # 残差连接
            
            # 8. 分类
            match_logits = self.classifier(fused_features)  # (batch_size, 1)
            match_probs.append(match_logits)
        
        # 平均多次采样的结果
        final_match_logits = torch.stack(match_probs, dim=0).mean(dim=0)
        
        return final_match_logits

# 测试代码
def test_improved_network():
    """测试改进后的网络架构"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建网络
    net = EdgeSparkNet(
        segment_length=32,
        n1=16,
        n2=16,
        feature_dim=256,
        hidden_channels=64,
        num_samples=3  # 测试时使用较少采样次数
    ).to(device)
    
    # 创建测试数据
    batch_size = 4
    edge_points1 = torch.randn(batch_size, 900, 2).to(device)
    edge_points2 = torch.randn(batch_size, 1000, 2).to(device)
    
    # 前向传播
    print("开始前向传播...")
    with torch.no_grad():
        match_prob = net(edge_points1, edge_points2)
    
    print(f"网络输出形状: {match_prob.shape}")
    print(f"网络输出: {match_prob.squeeze()}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试梯度计算
    print("\n测试梯度计算...")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 创建虚拟目标
    target = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    # 前向传播和反向传播
    match_prob = net(edge_points1, edge_points2)
    loss = F.binary_cross_entropy(match_prob, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"损失值: {loss.item():.4f}")
    print("梯度计算成功!")

if __name__ == "__main__":
    test_improved_network()