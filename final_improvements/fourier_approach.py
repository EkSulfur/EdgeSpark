#!/usr/bin/env python3
"""
基于傅里叶变换的碎片编码方法
用户提到的改进思路：基于傅里叶变换的碎片编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List

class FourierShapeEncoder(nn.Module):
    """
    基于傅里叶变换的形状编码器
    将边缘轮廓转换为频域特征
    """
    def __init__(self, max_points=1000, num_freqs=64, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.num_freqs = num_freqs
        self.feature_dim = feature_dim
        
        # 傅里叶特征处理网络 - 修复维度问题
        self.fourier_net = nn.Sequential(
            nn.Linear(num_freqs * 2 - 1, 256),  # real + imaginary parts (去掉一个相位)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # 空间特征编码器（作为补充）
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def extract_fourier_descriptors(self, points):
        """
        提取傅里叶描述符
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            fourier_features: (batch_size, num_freqs * 2)
        """
        batch_size, num_points, _ = points.shape
        
        # 转换为复数表示
        complex_points = points[:, :, 0] + 1j * points[:, :, 1]  # (batch_size, num_points)
        
        # 应用FFT
        fft_result = torch.fft.fft(complex_points, dim=1)  # (batch_size, num_points)
        
        # 取前num_freqs个频率分量
        if num_points > self.num_freqs:
            fft_result = fft_result[:, :self.num_freqs]
        else:
            # 如果点数不够，用零填充
            padding = torch.zeros(batch_size, self.num_freqs - num_points, 
                                device=fft_result.device, dtype=fft_result.dtype)
            fft_result = torch.cat([fft_result, padding], dim=1)
        
        # 分离实部和虚部
        real_part = fft_result.real
        imag_part = fft_result.imag
        
        # 组合特征
        fourier_features = torch.cat([real_part, imag_part], dim=1)  # (batch_size, num_freqs * 2)
        
        return fourier_features
        
    def extract_invariant_features(self, points):
        """
        提取旋转、平移、缩放不变的特征
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            invariant_features: (batch_size, feature_dim)
        """
        batch_size, num_points, _ = points.shape
        
        # 归一化到原点和单位尺度
        centroid = torch.mean(points, dim=1, keepdim=True)
        centered_points = points - centroid
        
        # 缩放归一化
        distances = torch.norm(centered_points, dim=2)
        max_dist = torch.max(distances, dim=1, keepdim=True)[0]
        max_dist = torch.clamp(max_dist, min=1e-6)
        normalized_points = centered_points / max_dist.unsqueeze(-1)
        
        # 傅里叶描述符
        fourier_desc = self.extract_fourier_descriptors(normalized_points)
        
        # 使傅里叶描述符具有旋转不变性
        # 方法：使用幅度谱而非相位
        real_part = fourier_desc[:, :self.num_freqs]
        imag_part = fourier_desc[:, self.num_freqs:]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        
        # 保留一些相位信息（相对相位）
        phase = torch.atan2(imag_part, real_part)
        relative_phase = phase[:, 1:] - phase[:, :-1]  # 相对相位
        
        # 组合幅度和相对相位
        combined_features = torch.cat([magnitude, relative_phase], dim=1)
        
        return combined_features
        
    def forward(self, points):
        """
        前向传播
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            shape_features: (batch_size, feature_dim)
        """
        # 1. 傅里叶特征
        invariant_features = self.extract_invariant_features(points)
        fourier_features = self.fourier_net(invariant_features)
        
        # 2. 空间特征（作为补充）
        spatial_input = points.transpose(1, 2)  # (batch_size, 2, num_points)
        spatial_features = self.spatial_encoder(spatial_input).squeeze(-1)
        
        # 3. 特征融合
        combined = torch.cat([fourier_features, spatial_features], dim=1)
        final_features = self.fusion(combined)
        
        return final_features

class FourierBasedMatchingNet(nn.Module):
    """
    基于傅里叶变换的碎片匹配网络
    """
    def __init__(self, max_points=1000, num_freqs=64, feature_dim=128):
        super().__init__()
        self.max_points = max_points
        self.feature_dim = feature_dim
        
        # 傅里叶形状编码器
        self.shape_encoder = FourierShapeEncoder(max_points, num_freqs, feature_dim)
        
        # 几何特征提取器
        self.geometry_extractor = nn.Sequential(
            nn.Linear(8, 64),  # 几何特征
            nn.ReLU(),
            nn.Linear(64, feature_dim // 4)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            nn.Linear(feature_dim * 4 + 2, 256),  # feat1, feat2, diff, hadamard, cosine + L2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def extract_geometric_features(self, points):
        """
        提取基本几何特征
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            geo_features: (batch_size, 8)
        """
        batch_size = points.shape[0]
        features = []
        
        for b in range(batch_size):
            pts = points[b].cpu().numpy()
            
            # 基本统计特征
            centroid = np.mean(pts, axis=0)
            std = np.std(pts, axis=0)
            
            # 边界框
            bbox_min = np.min(pts, axis=0)
            bbox_max = np.max(pts, axis=0)
            bbox_size = bbox_max - bbox_min
            
            # 组合特征
            geo_feat = np.concatenate([
                centroid,     # 2
                std,         # 2
                bbox_size,   # 2
                [np.mean(np.linalg.norm(pts - centroid, axis=1))],  # 1: 平均距离
                [np.std(np.linalg.norm(pts - centroid, axis=1))]    # 1: 距离标准差
            ])
            
            features.append(geo_feat)
        
        return torch.FloatTensor(features).to(points.device)
        
    def forward(self, points1, points2):
        """
        前向传播
        Args:
            points1: (batch_size, num_points1, 2)
            points2: (batch_size, num_points2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        # 1. 形状特征编码
        shape_feat1 = self.shape_encoder(points1)
        shape_feat2 = self.shape_encoder(points2)
        
        # 2. 几何特征提取
        geo_feat1 = self.extract_geometric_features(points1)
        geo_feat2 = self.extract_geometric_features(points2)
        
        geo_encoded1 = self.geometry_extractor(geo_feat1)
        geo_encoded2 = self.geometry_extractor(geo_feat2)
        
        # 3. 特征融合
        fused_feat1 = self.feature_fusion(torch.cat([shape_feat1, geo_encoded1], dim=1))
        fused_feat2 = self.feature_fusion(torch.cat([shape_feat2, geo_encoded2], dim=1))
        
        # 4. 匹配特征计算
        diff = fused_feat1 - fused_feat2
        hadamard = fused_feat1 * fused_feat2
        
        # 相似度度量
        cosine_sim = F.cosine_similarity(fused_feat1, fused_feat2, dim=1, eps=1e-8)
        l2_dist = torch.norm(diff, p=2, dim=1)
        
        # 5. 特征组合
        combined = torch.cat([
            fused_feat1,
            fused_feat2, 
            diff,
            hadamard,
            cosine_sim.unsqueeze(1),
            l2_dist.unsqueeze(1)
        ], dim=1)
        
        # 6. 匹配预测
        match_logits = self.matching_net(combined)
        
        return match_logits

class HybridFourierNet(nn.Module):
    """
    混合傅里叶网络：结合傅里叶变换和高采样方法
    """
    def __init__(self, max_points=1000, num_freqs=64, feature_dim=128, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        
        # 傅里叶编码器
        self.fourier_encoder = FourierShapeEncoder(max_points, num_freqs, feature_dim)
        
        # 多尺度采样器
        self.segment_lengths = [30, 50, 80]  # 多种段落长度
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            nn.Linear(feature_dim * 4 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 结果聚合
        self.aggregator = nn.Sequential(
            nn.Linear(len(self.segment_lengths), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def random_segment_sampling(self, points, segment_length, num_samples):
        """
        随机段落采样
        """
        batch_size, num_points, _ = points.shape
        
        if num_points < segment_length:
            return [points] * num_samples
            
        segments = []
        for _ in range(num_samples):
            start_indices = torch.randint(0, num_points - segment_length + 1, (batch_size,))
            
            segment_batch = []
            for b in range(batch_size):
                start_idx = start_indices[b].item()
                end_idx = start_idx + segment_length
                segment = points[b, start_idx:end_idx]
                segment_batch.append(segment)
            
            segments.append(torch.stack(segment_batch, dim=0))
        
        return segments
        
    def forward(self, points1, points2):
        """
        前向传播
        """
        scale_results = []
        
        # 对每种尺度进行处理
        for segment_length in self.segment_lengths:
            # 采样段落
            segments1 = self.random_segment_sampling(points1, segment_length, self.num_samples)
            segments2 = self.random_segment_sampling(points2, segment_length, self.num_samples)
            
            sample_scores = []
            
            # 处理每个采样对
            for seg1, seg2 in zip(segments1, segments2):
                # 傅里叶编码
                feat1 = self.fourier_encoder(seg1)
                feat2 = self.fourier_encoder(seg2)
                
                # 匹配特征
                diff = feat1 - feat2
                hadamard = feat1 * feat2
                cosine_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
                
                combined = torch.cat([feat1, feat2, diff, hadamard, cosine_sim.unsqueeze(1)], dim=1)
                score = self.matching_net(combined)
                sample_scores.append(score)
            
            # 平均采样结果
            scale_score = torch.mean(torch.stack(sample_scores, dim=1), dim=1)
            scale_results.append(scale_score)
        
        # 聚合多尺度结果
        scale_features = torch.cat(scale_results, dim=1)
        final_logits = self.aggregator(scale_features)
        
        return final_logits

# 测试函数
def test_fourier_networks():
    """测试傅里叶变换网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 测试基础傅里叶网络
    print("🧪 测试基础傅里叶网络...")
    fourier_net = FourierBasedMatchingNet(
        max_points=1000, 
        num_freqs=64, 
        feature_dim=128
    ).to(device)
    
    # 测试混合傅里叶网络
    print("🧪 测试混合傅里叶网络...")
    hybrid_net = HybridFourierNet(
        max_points=1000,
        num_freqs=64,
        feature_dim=128,
        num_samples=5
    ).to(device)
    
    # 测试数据
    batch_size = 4
    points1 = torch.randn(batch_size, 800, 2).to(device)
    points2 = torch.randn(batch_size, 900, 2).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        output1 = fourier_net(points1, points2)
        output2 = hybrid_net(points1, points2)
    
    print(f"基础傅里叶网络输出: {output1.shape}")
    print(f"基础傅里叶网络参数: {sum(p.numel() for p in fourier_net.parameters()):,}")
    
    print(f"混合傅里叶网络输出: {output2.shape}")
    print(f"混合傅里叶网络参数: {sum(p.numel() for p in hybrid_net.parameters()):,}")
    
    print("测试成功!")
    
    return fourier_net, hybrid_net

if __name__ == "__main__":
    test_fourier_networks()