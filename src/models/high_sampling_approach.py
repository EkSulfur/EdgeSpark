#!/usr/bin/env python3
"""
基于用户反馈的高采样次数改进方案
解决两个关键问题：
1. 增大采样次数 - 基于两个碎片可拼接边缘部分占总边缘长度较小的考虑
2. 改进采样后的综合部分 - 避免有效采样被"稀释"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List

class EdgeSegmentSampler(nn.Module):
    """
    智能边缘段采样器
    基于边缘特征重要性进行采样
    """
    def __init__(self, segment_length=50, num_samples=20):
        super().__init__()
        self.segment_length = segment_length
        self.num_samples = num_samples
        
        # 重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def extract_edge_curvature(self, points):
        """
        提取边缘曲率信息用于重要性评估
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            curvature: (batch_size, num_points)
        """
        batch_size, num_points, _ = points.shape
        
        # 计算切线方向
        if num_points < 3:
            return torch.zeros(batch_size, num_points).to(points.device)
            
        # 向前差分和向后差分
        forward_diff = torch.roll(points, -1, dims=1) - points
        backward_diff = points - torch.roll(points, 1, dims=1)
        
        # 计算曲率（简化版本）
        cross_product = forward_diff[:, :, 0] * backward_diff[:, :, 1] - forward_diff[:, :, 1] * backward_diff[:, :, 0]
        curvature = torch.abs(cross_product)
        
        return curvature
        
    def smart_sampling(self, points):
        """
        基于重要性的智能采样
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            sampled_segments: List of (batch_size, segment_length, 2)
        """
        batch_size, num_points, _ = points.shape
        
        if num_points < self.segment_length:
            # 如果点数不够，直接返回原始点云
            return [points]
            
        # 计算重要性权重
        x = points.transpose(1, 2)  # (batch_size, 2, num_points)
        importance_scores = self.importance_net(x).squeeze(1)  # (batch_size, num_points)
        
        # 结合曲率信息
        curvature = self.extract_edge_curvature(points)
        combined_importance = importance_scores + 0.3 * curvature
        
        segments = []
        for _ in range(self.num_samples):
            # 基于重要性权重采样起始点
            probs = F.softmax(combined_importance, dim=1)
            
            # 确保采样点有足够的后续点
            valid_starts = num_points - self.segment_length
            if valid_starts <= 0:
                segments.append(points)
                continue
                
            # 限制采样范围
            probs = probs[:, :valid_starts]
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # 采样起始点
            start_indices = torch.multinomial(probs, 1).squeeze(1)  # (batch_size,)
            
            # 提取段落
            segment_batch = []
            for b in range(batch_size):
                start_idx = start_indices[b].item()
                end_idx = start_idx + self.segment_length
                segment = points[b, start_idx:end_idx]
                segment_batch.append(segment)
            
            segments.append(torch.stack(segment_batch, dim=0))
            
            # 降低已采样区域的重要性（避免重复采样）
            for b in range(batch_size):
                start_idx = start_indices[b].item()
                end_idx = min(start_idx + self.segment_length, num_points)
                combined_importance[b, start_idx:end_idx] *= 0.1
        
        return segments

class SampleQualityAssessor(nn.Module):
    """
    采样质量评估器
    评估每个采样的匹配质量
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, segment1, segment2):
        """
        评估两个段落的匹配质量
        Args:
            segment1: (batch_size, num_points, 2)
            segment2: (batch_size, num_points, 2)
        Returns:
            quality_score: (batch_size, 1)
        """
        # 特征编码
        x1 = segment1.transpose(1, 2)  # (batch_size, 2, num_points)
        x2 = segment2.transpose(1, 2)
        
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        
        # 全局特征
        global_feat1 = self.global_pool(feat1).squeeze(-1)
        global_feat2 = self.global_pool(feat2).squeeze(-1)
        
        # 质量评估
        combined = torch.cat([global_feat1, global_feat2], dim=1)
        quality = self.quality_net(combined)
        
        return quality

class AttentionAggregator(nn.Module):
    """
    基于注意力的采样结果聚合器
    避免有效采样被稀释
    """
    def __init__(self, feature_dim=128, num_samples=20):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        
        # 注意力机制
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),  # feature + quality score
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, sample_features, quality_scores):
        """
        聚合多个采样结果
        Args:
            sample_features: (batch_size, num_samples, feature_dim)
            quality_scores: (batch_size, num_samples, 1)
        Returns:
            final_logits: (batch_size, 1)
        """
        batch_size, num_samples, feature_dim = sample_features.shape
        
        # 计算注意力权重
        attention_input = torch.cat([sample_features, quality_scores], dim=2)
        attention_weights = self.attention_net(attention_input)  # (batch_size, num_samples, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权聚合
        aggregated_features = torch.sum(sample_features * attention_weights, dim=1)
        
        # 最终预测
        final_logits = self.classifier(aggregated_features)
        
        return final_logits, attention_weights

class HighSamplingEdgeMatchingNet(nn.Module):
    """
    高采样次数边缘匹配网络
    实现用户建议的两个改进方向
    """
    def __init__(self, segment_length=50, num_samples=20, feature_dim=128):
        super().__init__()
        self.segment_length = segment_length
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
        # 智能采样器
        self.sampler = EdgeSegmentSampler(segment_length, num_samples)
        
        # 质量评估器
        self.quality_assessor = SampleQualityAssessor(feature_dim)
        
        # 段落特征编码器
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 匹配特征计算
        self.match_encoder = nn.Sequential(
            nn.Linear(feature_dim * 4 + 1, 256),  # feat1, feat2, diff, hadamard, cosine
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
        
        # 注意力聚合器
        self.aggregator = AttentionAggregator(feature_dim, num_samples)
        
    def encode_segment(self, segment):
        """
        编码单个段落
        Args:
            segment: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, feature_dim)
        """
        x = segment.transpose(1, 2)  # (batch_size, 2, num_points)
        features = self.segment_encoder(x).squeeze(-1)
        return features
        
    def compute_match_features(self, feat1, feat2):
        """
        计算匹配特征
        Args:
            feat1: (batch_size, feature_dim)
            feat2: (batch_size, feature_dim)
        Returns:
            match_features: (batch_size, feature_dim)
        """
        # 基本组合
        diff = feat1 - feat2
        hadamard = feat1 * feat2
        
        # 相似度度量
        cosine_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
        
        # 组合特征
        combined = torch.cat([
            feat1, feat2, diff, hadamard, cosine_sim.unsqueeze(1)
        ], dim=1)
        
        # 编码匹配特征
        match_features = self.match_encoder(combined)
        
        return match_features
        
    def forward(self, points1, points2):
        """
        前向传播
        Args:
            points1: (batch_size, num_points1, 2)
            points2: (batch_size, num_points2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        # 1. 智能采样
        segments1 = self.sampler.smart_sampling(points1)
        segments2 = self.sampler.smart_sampling(points2)
        
        # 2. 处理每个采样对
        sample_features = []
        quality_scores = []
        
        for i in range(len(segments1)):
            seg1 = segments1[i]
            seg2 = segments2[i]
            
            # 编码段落特征
            feat1 = self.encode_segment(seg1)
            feat2 = self.encode_segment(seg2)
            
            # 计算匹配特征
            match_feat = self.compute_match_features(feat1, feat2)
            sample_features.append(match_feat)
            
            # 评估采样质量
            quality = self.quality_assessor(seg1, seg2)
            quality_scores.append(quality)
        
        # 3. 堆叠特征
        sample_features = torch.stack(sample_features, dim=1)  # (batch_size, num_samples, feature_dim)
        quality_scores = torch.stack(quality_scores, dim=1)    # (batch_size, num_samples, 1)
        
        # 4. 注意力聚合
        final_logits, attention_weights = self.aggregator(sample_features, quality_scores)
        
        return final_logits

# 测试函数
def test_high_sampling_network():
    """测试高采样网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建网络
    model = HighSamplingEdgeMatchingNet(
        segment_length=50,
        num_samples=20,
        feature_dim=128
    ).to(device)
    
    # 测试数据
    batch_size = 4
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
    
    return model

if __name__ == "__main__":
    test_high_sampling_network()