"""
EdgeSpark 混合方法网络
结合最佳算法和暴力采样策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple

class EdgeShapeEncoder(nn.Module):
    """
    边缘形状编码器 (从final_approach复用)
    """
    def __init__(self, max_points=1000):
        super().__init__()
        self.max_points = max_points
        
        # 1. 局部形状特征提取
        self.local_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 2. 全局形状特征提取
        self.global_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 3. 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. 最终投影
        self.final_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
    def forward(self, points):
        """
        编码边缘形状
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, 64)
        """
        # 转换维度
        x = points.transpose(1, 2)  # (batch_size, 2, num_points)
        
        # 局部特征提取
        local_features = self.local_conv(x)
        
        # 全局特征提取
        global_features = self.global_conv(local_features)
        
        # 自适应池化
        pooled = self.adaptive_pool(global_features).squeeze(-1)
        
        # 最终投影
        final_features = self.final_proj(pooled)
        
        return final_features

class MultiSampleStrategy:
    """
    多采样策略类
    """
    def __init__(self, num_samples=5, sample_method='diversified'):
        self.num_samples = num_samples
        self.sample_method = sample_method
        
    def diversified_sampling(self, points: torch.Tensor, max_points: int, seed: int = None) -> torch.Tensor:
        """
        多样化采样策略
        确保采样覆盖边缘的不同部分
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        N = points.shape[0]
        if N <= max_points:
            # 如果点数不足，重复采样
            indices = torch.randint(0, N, (max_points,), device=points.device)
            return points[indices]
        
        # 分段采样策略
        segment_size = max_points // 4  # 分成4段
        segments = []
        
        for i in range(4):
            start = i * N // 4
            end = (i + 1) * N // 4
            
            if end - start >= segment_size:
                # 在该段内随机采样
                segment_indices = torch.randint(start, end, (segment_size,), device=points.device)
            else:
                # 如果段太小，重复采样
                segment_indices = torch.randint(start, end, (segment_size,), device=points.device)
            
            segments.append(points[segment_indices])
        
        # 如果还有剩余点数需要采样
        remaining = max_points - segment_size * 4
        if remaining > 0:
            extra_indices = torch.randint(0, N, (remaining,), device=points.device)
            segments.append(points[extra_indices])
        
        return torch.cat(segments, dim=0)
    
    def random_sampling(self, points: torch.Tensor, max_points: int, seed: int = None) -> torch.Tensor:
        """
        纯随机采样
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        N = points.shape[0]
        if N <= max_points:
            indices = torch.randint(0, N, (max_points,), device=points.device)
        else:
            indices = torch.randint(0, N, (max_points,), device=points.device)
        
        return points[indices]
    
    def sample_points(self, points: torch.Tensor, max_points: int, sample_idx: int) -> torch.Tensor:
        """
        根据策略采样点
        """
        if self.sample_method == 'diversified':
            return self.diversified_sampling(points, max_points, seed=sample_idx)
        else:
            return self.random_sampling(points, max_points, seed=sample_idx)

class EnsembleStrategy:
    """
    集成策略类
    """
    def __init__(self, method='weighted_average'):
        self.method = method
        
    def simple_average(self, scores: List[torch.Tensor]) -> torch.Tensor:
        """简单平均"""
        return torch.mean(torch.stack(scores), dim=0)
    
    def weighted_average(self, scores: List[torch.Tensor], weights: torch.Tensor = None) -> torch.Tensor:
        """加权平均"""
        if weights is None:
            return self.simple_average(scores)
        
        stacked_scores = torch.stack(scores)  # (num_samples, batch_size, 1)
        weights = weights.unsqueeze(-1)  # (num_samples, 1)
        
        weighted_scores = stacked_scores * weights
        return torch.sum(weighted_scores, dim=0)
    
    def soft_voting(self, scores: List[torch.Tensor]) -> torch.Tensor:
        """软投票"""
        probs = [torch.sigmoid(score) for score in scores]
        avg_prob = torch.mean(torch.stack(probs), dim=0)
        return torch.logit(avg_prob.clamp(1e-7, 1-1e-7))  # 转换回logit
    
    def confidence_weighted(self, scores: List[torch.Tensor]) -> torch.Tensor:
        """基于置信度的加权"""
        # 计算每个样本的置信度 (基于sigmoid输出离0.5的距离)
        probs = [torch.sigmoid(score) for score in scores]
        confidences = [torch.abs(prob - 0.5) for prob in probs]
        
        # 归一化权重
        total_confidence = torch.sum(torch.stack(confidences), dim=0)
        weights = [conf / (total_confidence + 1e-7) for conf in confidences]
        
        # 加权平均
        weighted_scores = [score * weight for score, weight in zip(scores, weights)]
        return torch.sum(torch.stack(weighted_scores), dim=0)
    
    def ensemble(self, scores: List[torch.Tensor]) -> torch.Tensor:
        """根据策略集成分数"""
        if self.method == 'simple_average':
            return self.simple_average(scores)
        elif self.method == 'weighted_average':
            return self.weighted_average(scores)
        elif self.method == 'soft_voting':
            return self.soft_voting(scores)
        elif self.method == 'confidence_weighted':
            return self.confidence_weighted(scores)
        else:
            return self.simple_average(scores)

class HybridEdgeMatchingNet(nn.Module):
    """
    混合边缘匹配网络
    结合最佳算法和暴力采样
    """
    def __init__(self, 
                 max_points=1000,
                 num_samples=5,
                 sample_method='diversified',
                 ensemble_method='weighted_average'):
        super().__init__()
        self.max_points = max_points
        self.num_samples = num_samples
        
        # 形状编码器
        self.shape_encoder = EdgeShapeEncoder(max_points)
        
        # 采样策略
        self.sampler = MultiSampleStrategy(num_samples, sample_method)
        
        # 集成策略
        self.ensemble = EnsembleStrategy(ensemble_method)
        
        # 匹配网络
        self.matching_net = nn.Sequential(
            nn.Linear(64 * 2 + 64 + 1, 128),
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
    
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            points1: (batch_size, num_points1, 2)
            points2: (batch_size, num_points2, 2)
        Returns:
            match_logits: (batch_size, 1)
        """
        batch_size = points1.shape[0]
        device = points1.device
        
        # 多次采样的结果
        sample_scores = []
        
        for sample_idx in range(self.num_samples):
            # 对每个batch中的样本进行采样
            sampled_points1 = []
            sampled_points2 = []
            
            for b in range(batch_size):
                # 采样点云
                sample1 = self.sampler.sample_points(points1[b], self.max_points, sample_idx)
                sample2 = self.sampler.sample_points(points2[b], self.max_points, sample_idx)
                
                sampled_points1.append(sample1)
                sampled_points2.append(sample2)
            
            # 转换为tensor
            sampled_points1 = torch.stack(sampled_points1)  # (batch_size, max_points, 2)
            sampled_points2 = torch.stack(sampled_points2)  # (batch_size, max_points, 2)
            
            # 形状编码
            shape1 = self.shape_encoder(sampled_points1)  # (batch_size, 64)
            shape2 = self.shape_encoder(sampled_points2)  # (batch_size, 64)
            
            # 特征组合
            diff = shape1 - shape2
            dot = torch.sum(shape1 * shape2, dim=1, keepdim=True)
            combined = torch.cat([shape1, shape2, diff, dot], dim=1)
            
            # 匹配预测
            match_logits = self.matching_net(combined)
            sample_scores.append(match_logits)
        
        # 集成多次采样的结果
        final_logits = self.ensemble.ensemble(sample_scores)
        
        return final_logits

# 测试函数
def test_hybrid_network():
    """测试混合网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建网络
    model = HybridEdgeMatchingNet(
        max_points=1000,
        num_samples=3,  # 测试时使用较少采样
        sample_method='diversified',
        ensemble_method='weighted_average'
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

if __name__ == "__main__":
    test_hybrid_network()