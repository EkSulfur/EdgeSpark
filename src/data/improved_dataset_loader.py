import torch
import torch.utils.data as data
import pickle
import numpy as np
from typing import Dict, List, Tuple
import random

class ImprovedEdgeSparkDataset(data.Dataset):
    """
    改进版EdgeSpark数据集加载器
    主要改进：保持边缘点云的有序性，避免随机采样导致的空间连续性丢失
    """
    def __init__(self, pkl_path: str, max_points: int = 2000, augment: bool = True, sampling_strategy: str = 'ordered'):
        """
        初始化数据集
        Args:
            pkl_path: pickle文件路径
            max_points: 最大点数限制
            augment: 是否使用数据增强
            sampling_strategy: 采样策略 ('ordered', 'random', 'padding')
        """
        self.pkl_path = pkl_path
        self.max_points = max_points
        self.augment = augment
        self.sampling_strategy = sampling_strategy
        
        # 加载数据
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 解析数据
        self.edge_points = self.data['full_pcd_all']  # 边缘点云列表
        self.gt_pairs = self.data['GT_pairs']  # 匹配对
        self.source_indices = self.data['source_ind']  # 源碎片匹配点索引
        self.target_indices = self.data['target_ind']  # 目标碎片匹配点索引
        
        # 创建正样本和负样本
        self.samples = self._create_samples()
        
        print(f"改进数据集加载完成: {len(self.samples)} 个样本 (策略: {sampling_strategy})")
        print(f"正样本: {sum(1 for s in self.samples if s['label'] == 1)}")
        print(f"负样本: {sum(1 for s in self.samples if s['label'] == 0)}")
    
    def _create_samples(self) -> List[Dict]:
        """创建训练样本（正样本和负样本）"""
        samples = []
        
        # 1. 创建正样本
        for i, (source_idx, target_idx) in enumerate(self.gt_pairs):
            samples.append({
                'source_idx': source_idx,
                'target_idx': target_idx,
                'label': 1,
                'source_match_points': self.source_indices[i],
                'target_match_points': self.target_indices[i]
            })
        
        # 2. 创建负样本（随机配对）
        num_fragments = len(self.edge_points)
        num_negative = len(self.gt_pairs)  # 负样本数量等于正样本数量
        
        # 创建所有可能的配对
        all_pairs = set()
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                all_pairs.add((i, j))
        
        # 移除正样本对
        positive_pairs = set()
        for source_idx, target_idx in self.gt_pairs:
            pair = (min(source_idx, target_idx), max(source_idx, target_idx))
            positive_pairs.add(pair)
        
        negative_pairs = list(all_pairs - positive_pairs)
        
        # 随机选择负样本
        selected_negative = random.sample(negative_pairs, min(num_negative, len(negative_pairs)))
        
        for source_idx, target_idx in selected_negative:
            samples.append({
                'source_idx': source_idx,
                'target_idx': target_idx,
                'label': 0,
                'source_match_points': None,
                'target_match_points': None
            })
        
        return samples
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """归一化点云到[-1, 1]范围"""
        if len(points) == 0:
            return points
        
        # 计算边界框
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 避免除零
        range_coords = max_coords - min_coords
        range_coords = np.where(range_coords == 0, 1, range_coords)
        # 添加小的epsilon避免数值不稳定
        range_coords = np.maximum(range_coords, 1e-8)
        
        # 归一化到[-1, 1]
        normalized = 2 * (points - min_coords) / range_coords - 1
        
        return normalized
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        """数据增强：旋转、缩放、噪声"""
        if not self.augment:
            return points
        
        # 随机旋转
        if random.random() < 0.5:
            angle = random.uniform(-np.pi/6, np.pi/6)  # ±30度
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points = points @ rotation_matrix.T
        
        # 随机缩放
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            points = points * scale
        
        # 添加噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, points.shape)
            points = points + noise
        
        return points
    
    def _pad_or_sample_points_ordered(self, points: np.ndarray) -> np.ndarray:
        """改进版：保持有序性的点云处理"""
        if len(points) >= self.max_points:
            # 等间隔采样，保持边缘点的空间连续性
            step = len(points) / self.max_points
            indices = np.round(np.arange(0, len(points), step)[:self.max_points]).astype(int)
            # 确保不越界
            indices = np.clip(indices, 0, len(points) - 1)
            return points[indices]
        else:
            # 用特殊标记填充，保持原始点的有序性
            padding = np.full((self.max_points - len(points), 2), -999.0)  # 使用特殊值标记padding
            return np.vstack([points, padding])
    
    def _pad_or_sample_points_padding(self, points: np.ndarray) -> np.ndarray:
        """纯填充策略：用零填充"""
        if len(points) >= self.max_points:
            # 简单截断到前max_points个点
            return points[:self.max_points]
        else:
            # 用零填充
            padding = np.zeros((self.max_points - len(points), 2))
            return np.vstack([points, padding])
    
    def _pad_or_sample_points_random(self, points: np.ndarray) -> np.ndarray:
        """原始随机采样策略（用于对比）"""
        if len(points) >= self.max_points:
            # 随机采样
            indices = np.random.choice(len(points), self.max_points, replace=False)
            return points[indices]
        else:
            # 重复采样
            indices = np.random.choice(len(points), self.max_points, replace=True)
            return points[indices]
    
    def _process_points(self, points: np.ndarray) -> np.ndarray:
        """根据策略处理点云"""
        if self.sampling_strategy == 'ordered':
            return self._pad_or_sample_points_ordered(points)
        elif self.sampling_strategy == 'padding':
            return self._pad_or_sample_points_padding(points)
        elif self.sampling_strategy == 'random':
            return self._pad_or_sample_points_random(points)
        else:
            raise ValueError(f"未知的采样策略: {self.sampling_strategy}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 获取边缘点云
        source_points = self.edge_points[sample['source_idx']].copy()
        target_points = self.edge_points[sample['target_idx']].copy()
        
        # 数据增强
        source_points = self._augment_points(source_points)
        target_points = self._augment_points(target_points)
        
        # 归一化
        source_points = self._normalize_points(source_points)
        target_points = self._normalize_points(target_points)
        
        # 根据策略处理点云
        source_points = self._process_points(source_points)
        target_points = self._process_points(target_points)
        
        # 转换为Tensor
        return {
            'source_points': torch.FloatTensor(source_points),
            'target_points': torch.FloatTensor(target_points),
            'label': torch.FloatTensor([sample['label']]),
            'source_idx': sample['source_idx'],
            'target_idx': sample['target_idx']
        }

def create_improved_dataloaders(train_pkl: str, 
                               valid_pkl: str, 
                               test_pkl: str,
                               batch_size: int = 32,
                               max_points: int = 2000,
                               num_workers: int = 4,
                               sampling_strategy: str = 'ordered') -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    创建改进版数据加载器
    """
    # 创建数据集
    train_dataset = ImprovedEdgeSparkDataset(train_pkl, max_points=max_points, augment=True, sampling_strategy=sampling_strategy)
    valid_dataset = ImprovedEdgeSparkDataset(valid_pkl, max_points=max_points, augment=False, sampling_strategy=sampling_strategy)
    test_dataset = ImprovedEdgeSparkDataset(test_pkl, max_points=max_points, augment=False, sampling_strategy=sampling_strategy)
    
    # 创建数据加载器
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader
