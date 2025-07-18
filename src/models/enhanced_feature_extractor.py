"""
增强的几何特征提取器
专门设计用于解决特征可分离性不足的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist

class GeometricFeatureExtractor:
    """几何特征提取器"""
    
    def __init__(self):
        self.eps = 1e-8
    
    def extract_curvature_features(self, points: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        提取曲率特征
        Args:
            points: (N, 2) 边缘点坐标
            window_size: 计算曲率的窗口大小
        Returns:
            curvature_features: 曲率相关特征
        """
        if len(points) < window_size:
            return np.zeros(6)
        
        # 计算局部曲率
        curvatures = []
        for i in range(len(points)):
            # 获取局部窗口
            indices = [(i + j - window_size//2) % len(points) for j in range(window_size)]
            local_points = points[indices]
            
            # 拟合圆弧计算曲率
            if len(local_points) >= 3:
                curvature = self._compute_local_curvature(local_points)
                curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # 曲率统计特征
        features = [
            np.mean(curvatures),           # 平均曲率
            np.std(curvatures),            # 曲率标准差
            np.max(curvatures),            # 最大曲率
            np.min(curvatures),            # 最小曲率
            np.sum(curvatures > np.median(curvatures)) / len(curvatures),  # 高曲率比例
            np.sum(np.abs(np.diff(curvatures)))  # 曲率变化总量
        ]
        
        return np.array(features)
    
    def _compute_local_curvature(self, points: np.ndarray) -> float:
        """计算局部曲率（使用三点法）"""
        if len(points) < 3:
            return 0.0
        
        # 选择三个点
        p1, p2, p3 = points[0], points[len(points)//2], points[-1]
        
        # 计算三角形面积
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        
        # 计算三边长度
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # 外接圆半径
        if area < self.eps:
            return 0.0
        
        R = (a * b * c) / (4 * area + self.eps)
        
        # 曲率是半径的倒数
        return 1.0 / (R + self.eps)
    
    def extract_angle_features(self, points: np.ndarray) -> np.ndarray:
        """
        提取角度特征
        Args:
            points: (N, 2) 边缘点坐标
        Returns:
            angle_features: 角度相关特征
        """
        if len(points) < 3:
            return np.zeros(8)
        
        # 计算相邻线段角度
        angles = []
        for i in range(len(points)):
            p1 = points[i-1]
            p2 = points[i]
            p3 = points[(i+1) % len(points)]
            
            # 计算两个向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算角度
            angle = self._compute_angle(v1, v2)
            angles.append(angle)
        
        angles = np.array(angles)
        
        # 角度统计特征
        features = [
            np.mean(angles),               # 平均角度
            np.std(angles),                # 角度标准差
            np.sum(angles < np.pi/2) / len(angles),  # 锐角比例
            np.sum(angles > 3*np.pi/2) / len(angles),  # 凹角比例
            np.max(angles),                # 最大角度
            np.min(angles),                # 最小角度
            np.sum(np.abs(np.diff(angles))),  # 角度变化总量
            len(self._find_corner_points(points, angles))  # 角点数量
        ]
        
        return np.array(features)
    
    def _compute_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量之间的角度"""
        len1 = np.linalg.norm(v1) + self.eps
        len2 = np.linalg.norm(v2) + self.eps
        
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle)
    
    def _find_corner_points(self, points: np.ndarray, angles: np.ndarray, threshold: float = np.pi/3) -> List[int]:
        """找到角点（角度变化大的点）"""
        corners = []
        for i, angle in enumerate(angles):
            if angle < threshold or angle > 2*np.pi - threshold:
                corners.append(i)
        return corners
    
    def extract_fourier_descriptors(self, points: np.ndarray, n_descriptors: int = 16) -> np.ndarray:
        """
        提取傅里叶描述子
        Args:
            points: (N, 2) 边缘点坐标
            n_descriptors: 傅里叶描述子数量
        Returns:
            fourier_features: 傅里叶描述子特征
        """
        if len(points) < 4:
            return np.zeros(n_descriptors)
        
        # 将点转换为复数表示
        complex_points = points[:, 0] + 1j * points[:, 1]
        
        # 计算傅里叶变换
        fft_coeffs = np.fft.fft(complex_points)
        
        # 归一化（平移不变性）
        if abs(fft_coeffs[0]) > self.eps:
            fft_coeffs = fft_coeffs / fft_coeffs[0]
        
        # 旋转不变性（使用幅值）
        magnitudes = np.abs(fft_coeffs)
        
        # 尺度不变性（归一化）
        if magnitudes[1] > self.eps:
            magnitudes = magnitudes / magnitudes[1]
        
        # 取前n_descriptors个描述子
        n_take = min(n_descriptors, len(magnitudes))
        descriptors = magnitudes[1:n_take+1]  # 跳过直流分量
        
        # 如果不够，用零填充
        if len(descriptors) < n_descriptors:
            descriptors = np.pad(descriptors, (0, n_descriptors - len(descriptors)))
        
        return descriptors
    
    def extract_geometric_moments(self, points: np.ndarray) -> np.ndarray:
        """
        提取几何矩特征
        Args:
            points: (N, 2) 边缘点坐标
        Returns:
            moment_features: 几何矩特征
        """
        if len(points) < 3:
            return np.zeros(7)
        
        # 计算质心
        centroid = np.mean(points, axis=0)
        
        # 中心化
        centered_points = points - centroid
        
        # 计算各阶矩
        x, y = centered_points[:, 0], centered_points[:, 1]
        
        # 二阶矩
        m20 = np.mean(x**2)
        m02 = np.mean(y**2)
        m11 = np.mean(x * y)
        
        # 计算主轴方向
        if m20 != m02:
            theta = 0.5 * np.arctan(2 * m11 / (m20 - m02 + self.eps))
        else:
            theta = np.pi / 4 if m11 > 0 else -np.pi / 4
        
        # 计算椭圆参数
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        # 主轴上的矩
        mu20_prime = m20 * cos_theta**2 + m02 * sin_theta**2 + 2 * m11 * cos_theta * sin_theta
        mu02_prime = m20 * sin_theta**2 + m02 * cos_theta**2 - 2 * m11 * cos_theta * sin_theta
        
        # 计算特征
        features = [
            m20 + m02,                     # 总惯性
            abs(m20 - m02),                # 惯性差
            abs(m11),                      # 偏斜度
            theta,                         # 主轴角度
            mu20_prime / (mu02_prime + self.eps),  # 长短轴比
            np.sqrt(mu20_prime),           # 主轴长度
            np.sqrt(mu02_prime)            # 次轴长度
        ]
        
        return np.array(features)
    
    def extract_distance_features(self, points: np.ndarray) -> np.ndarray:
        """
        提取距离特征
        Args:
            points: (N, 2) 边缘点坐标
        Returns:
            distance_features: 距离相关特征
        """
        if len(points) < 2:
            return np.zeros(6)
        
        # 计算质心
        centroid = np.mean(points, axis=0)
        
        # 到质心的距离
        distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
        
        # 相邻点距离
        adjacent_distances = []
        for i in range(len(points)):
            dist = np.linalg.norm(points[i] - points[(i+1) % len(points)])
            adjacent_distances.append(dist)
        adjacent_distances = np.array(adjacent_distances)
        
        features = [
            np.mean(distances_to_centroid),     # 平均半径
            np.std(distances_to_centroid),      # 半径标准差
            np.max(distances_to_centroid),      # 最大半径
            np.mean(adjacent_distances),        # 平均边长
            np.std(adjacent_distances),         # 边长标准差
            np.max(adjacent_distances) / (np.mean(adjacent_distances) + self.eps)  # 边长变化度
        ]
        
        return np.array(features)
    
    def extract_all_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """提取所有几何特征"""
        features = {
            'curvature': self.extract_curvature_features(points),
            'angle': self.extract_angle_features(points),
            'fourier': self.extract_fourier_descriptors(points),
            'moment': self.extract_geometric_moments(points),
            'distance': self.extract_distance_features(points)
        }
        
        return features

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, max_points: int = 1000):
        super().__init__()
        self.max_points = max_points
        self.geometric_extractor = GeometricFeatureExtractor()
        
        # 几何特征维度
        self.curvature_dim = 6
        self.angle_dim = 8
        self.fourier_dim = 16
        self.moment_dim = 7
        self.distance_dim = 6
        self.total_geometric_dim = self.curvature_dim + self.angle_dim + self.fourier_dim + self.moment_dim + self.distance_dim
        
        # 原始坐标特征提取器（保留但改进）
        self.coordinate_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 几何特征编码器
        self.geometric_encoder = nn.Sequential(
            nn.Linear(self.total_geometric_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # 特征归一化
        self.feature_norm = nn.LayerNorm(128)
        
    def extract_geometric_features_batch(self, points_batch: torch.Tensor) -> torch.Tensor:
        """批量提取几何特征"""
        batch_size = points_batch.shape[0]
        geometric_features = torch.zeros(batch_size, self.total_geometric_dim, device=points_batch.device)
        
        for i in range(batch_size):
            # 转换为numpy进行几何特征提取
            points_np = points_batch[i].cpu().numpy()
            
            # 去除padding（假设padding值为-999）
            valid_mask = ~((points_np == -999).all(axis=1))
            if valid_mask.sum() > 0:
                valid_points = points_np[valid_mask]
                
                # 提取所有几何特征
                features = self.geometric_extractor.extract_all_features(valid_points)
                
                # 拼接所有特征
                feature_vector = np.concatenate([
                    features['curvature'],
                    features['angle'],
                    features['fourier'],
                    features['moment'],
                    features['distance']
                ])
                
                # 处理NaN和Inf
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
                
                geometric_features[i] = torch.FloatTensor(feature_vector).to(points_batch.device)
        
        return geometric_features
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            points: (batch_size, num_points, 2)
        Returns:
            features: (batch_size, 128)
        """
        batch_size = points.shape[0]
        
        # 1. 提取几何特征
        geometric_features = self.extract_geometric_features_batch(points)
        
        # 2. 提取坐标特征
        coord_input = points.transpose(1, 2)  # (batch_size, 2, num_points)
        coordinate_features = self.coordinate_encoder(coord_input).squeeze(-1)  # (batch_size, 256)
        
        # 3. 编码几何特征
        encoded_geometric = self.geometric_encoder(geometric_features)  # (batch_size, 256)
        
        # 4. 特征融合
        combined_features = torch.cat([coordinate_features, encoded_geometric], dim=1)
        fused_features = self.feature_fusion(combined_features)  # (batch_size, 128)
        
        # 5. 特征归一化
        normalized_features = self.feature_norm(fused_features)
        
        return normalized_features

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        Args:
            features1: (batch_size, feature_dim) 第一组特征
            features2: (batch_size, feature_dim) 第二组特征
            labels: (batch_size,) 标签 (1表示匹配, 0表示不匹配)
        """
        # 计算余弦相似度
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)
        similarity = torch.sum(features1_norm * features2_norm, dim=1)
        
        # 对比损失
        positive_mask = labels.float()
        negative_mask = 1 - positive_mask
        
        # 正样本损失：最大化相似度
        positive_loss = positive_mask * (1 - similarity)
        
        # 负样本损失：最小化相似度（但有margin）
        negative_loss = negative_mask * torch.clamp(similarity - self.margin, min=0)
        
        loss = positive_loss + negative_loss
        
        return loss.mean()

class EnhancedFragmentMatcher(nn.Module):
    """增强的碎片匹配网络"""
    
    def __init__(self, max_points: int = 1000):
        super().__init__()
        self.max_points = max_points
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(max_points)
        
        # 匹配网络
        self.matcher = nn.Sequential(
            # 输入：两个128维特征 + 差值 + 相似度
            nn.Linear(128 * 2 + 128 + 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            points1: (batch_size, num_points, 2)
            points2: (batch_size, num_points, 2)
        Returns:
            match_logits: (batch_size, 1) 匹配概率
            features1: (batch_size, 128) 第一组特征
            features2: (batch_size, 128) 第二组特征
        """
        # 提取特征
        features1 = self.feature_extractor(points1)
        features2 = self.feature_extractor(points2)
        
        # 计算特征关系
        diff = features1 - features2  # 差值特征
        
        # 多种相似度度量
        cosine_sim = F.cosine_similarity(features1, features2, dim=1, eps=1e-8).unsqueeze(1)
        euclidean_dist = torch.norm(features1 - features2, p=2, dim=1, keepdim=True)
        dot_product = torch.sum(features1 * features2, dim=1, keepdim=True)
        
        # 拼接所有特征
        combined = torch.cat([
            features1, features2, diff, 
            cosine_sim, euclidean_dist, dot_product
        ], dim=1)
        
        # 匹配预测
        match_logits = self.matcher(combined)
        
        return match_logits, features1, features2

def test_geometric_feature_extractor():
    """测试几何特征提取器"""
    print("🧪 测试几何特征提取器...")
    
    # 创建测试数据
    # 圆形
    t = np.linspace(0, 2*np.pi, 100)
    circle_points = np.column_stack([np.cos(t), np.sin(t)])
    
    # 正方形
    square_points = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ])
    
    extractor = GeometricFeatureExtractor()
    
    # 测试圆形特征
    print("\n圆形特征:")
    circle_features = extractor.extract_all_features(circle_points)
    for name, features in circle_features.items():
        print(f"  {name}: {features[:3]}...")  # 只显示前3个值
    
    # 测试正方形特征
    print("\n正方形特征:")
    square_features = extractor.extract_all_features(square_points)
    for name, features in square_features.items():
        print(f"  {name}: {features[:3]}...")
    
    print("✅ 几何特征提取器测试完成")

if __name__ == "__main__":
    test_geometric_feature_extractor()