import torch
import torch.utils.data as data
import pickle
import numpy as np
from typing import Dict, List, Tuple
import random
from sklearn.metrics.pairwise import euclidean_distances

class SimpleEdgeSparkDataset(data.Dataset):
    """
    简化版EdgeSpark数据集
    改进负采样策略和数据处理
    """
    def __init__(self, pkl_path: str, max_points: int = 1000, augment: bool = True, 
                 negative_ratio: float = 1.0, hard_negative_ratio: float = 0.3):
        """
        初始化数据集
        Args:
            pkl_path: pickle文件路径
            max_points: 最大点数限制
            augment: 是否数据增强
            negative_ratio: 负样本比例
            hard_negative_ratio: 困难负样本比例
        """
        self.pkl_path = pkl_path
        self.max_points = max_points
        self.augment = augment
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        
        # 加载数据
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 解析数据
        self.edge_points = self.data['full_pcd_all']
        self.gt_pairs = self.data['GT_pairs']
        self.source_indices = self.data['source_ind']
        self.target_indices = self.data['target_ind']
        
        # 预计算碎片特征用于困难负样本挖掘
        self.fragment_features = self._compute_fragment_features()
        
        # 创建样本
        self.samples = self._create_balanced_samples()
        
        print(f"数据集: {len(self.samples)} 样本")
        print(f"正样本: {sum(1 for s in self.samples if s['label'] == 1)}")
        print(f"负样本: {sum(1 for s in self.samples if s['label'] == 0)}")
        
    def _compute_fragment_features(self) -> List[np.ndarray]:
        """计算每个碎片的简单特征用于困难负样本挖掘"""
        features = []
        
        for points in self.edge_points:
            if len(points) == 0:
                features.append(np.zeros(6))
                continue
                
            # 简单几何特征
            centroid = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            
            feature = np.concatenate([centroid, std, bbox_min, bbox_max])
            features.append(feature)
            
        return features
    
    def _create_balanced_samples(self) -> List[Dict]:
        """创建平衡的训练样本"""
        samples = []
        
        # 1. 添加所有正样本
        for i, (source_idx, target_idx) in enumerate(self.gt_pairs):
            samples.append({
                'source_idx': source_idx,
                'target_idx': target_idx,
                'label': 1,
                'source_match_points': self.source_indices[i],
                'target_match_points': self.target_indices[i]
            })
        
        # 2. 生成负样本
        num_positive = len(self.gt_pairs)
        num_negative = int(num_positive * self.negative_ratio)
        
        # 创建正样本对集合
        positive_pairs = set()
        for source_idx, target_idx in self.gt_pairs:
            positive_pairs.add((min(source_idx, target_idx), max(source_idx, target_idx)))
        
        # 生成所有可能的负样本对
        num_fragments = len(self.edge_points)
        all_possible_pairs = []
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                if (i, j) not in positive_pairs:
                    all_possible_pairs.append((i, j))
        
        # 分为困难负样本和随机负样本
        num_hard_negative = int(num_negative * self.hard_negative_ratio)
        num_random_negative = num_negative - num_hard_negative
        
        # 3. 困难负样本挖掘
        hard_negatives = self._mine_hard_negatives(all_possible_pairs, num_hard_negative)
        
        # 4. 随机负样本
        remaining_pairs = [p for p in all_possible_pairs if p not in hard_negatives]
        random_negatives = random.sample(remaining_pairs, 
                                       min(num_random_negative, len(remaining_pairs)))
        
        # 添加负样本
        for source_idx, target_idx in hard_negatives + random_negatives:
            samples.append({
                'source_idx': source_idx,
                'target_idx': target_idx,
                'label': 0,
                'source_match_points': None,
                'target_match_points': None
            })
        
        return samples
    
    def _mine_hard_negatives(self, candidate_pairs: List[Tuple], num_hard: int) -> List[Tuple]:
        """挖掘困难负样本"""
        if num_hard == 0:
            return []
        
        # 计算候选对的特征距离
        pair_distances = []
        for i, j in candidate_pairs:
            feat_i = self.fragment_features[i]
            feat_j = self.fragment_features[j]
            dist = np.linalg.norm(feat_i - feat_j)
            pair_distances.append((dist, (i, j)))
        
        # 选择距离最小的作为困难负样本
        pair_distances.sort(key=lambda x: x[0])
        hard_negatives = [pair for _, pair in pair_distances[:num_hard]]
        
        return hard_negatives
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """稳定的点云归一化"""
        if len(points) == 0:
            return points
        
        # 计算边界框
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 计算范围，避免除零
        range_coords = max_coords - min_coords
        range_coords = np.maximum(range_coords, 1e-6)  # 避免除零
        
        # 归一化到[0, 1]然后平移到[-1, 1]
        normalized = (points - min_coords) / range_coords
        normalized = 2 * normalized - 1
        
        return normalized
    
    def _light_augment(self, points: np.ndarray) -> np.ndarray:
        """轻量级数据增强"""
        if not self.augment:
            return points
        
        # 只做轻微的随机变换
        if random.random() < 0.3:
            # 小幅旋转
            angle = random.uniform(-np.pi/12, np.pi/12)  # ±15度
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points = points @ rotation_matrix.T
        
        if random.random() < 0.2:
            # 小幅缩放
            scale = random.uniform(0.95, 1.05)
            points = points * scale
        
        if random.random() < 0.2:
            # 小幅噪声
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
        
        return points
    
    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """智能点采样"""
        if len(points) == 0:
            return np.zeros((self.max_points, 2))
        
        if len(points) <= self.max_points:
            # 重复采样
            indices = np.random.choice(len(points), self.max_points, replace=True)
            return points[indices]
        else:
            # 降采样 - 使用分层采样保持分布
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=int)
            # 添加一些随机性
            noise = np.random.randint(-2, 3, size=self.max_points)
            indices = np.clip(indices + noise, 0, len(points) - 1)
            return points[indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        sample = self.samples[idx]
        
        # 获取点云
        source_points = self.edge_points[sample['source_idx']].copy()
        target_points = self.edge_points[sample['target_idx']].copy()
        
        # 数据增强
        source_points = self._light_augment(source_points)
        target_points = self._light_augment(target_points)
        
        # 归一化
        source_points = self._normalize_points(source_points)
        target_points = self._normalize_points(target_points)
        
        # 采样固定点数
        source_points = self._sample_points(source_points)
        target_points = self._sample_points(target_points)
        
        return {
            'source_points': torch.FloatTensor(source_points),
            'target_points': torch.FloatTensor(target_points),
            'label': torch.FloatTensor([sample['label']]),
            'source_idx': sample['source_idx'],
            'target_idx': sample['target_idx']
        }

def create_simple_dataloaders(train_pkl: str, 
                            valid_pkl: str, 
                            test_pkl: str,
                            batch_size: int = 32,
                            max_points: int = 1000,
                            num_workers: int = 4) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """创建简化版数据加载器"""
    
    # 创建数据集
    train_dataset = SimpleEdgeSparkDataset(
        train_pkl, 
        max_points=max_points, 
        augment=True,
        negative_ratio=1.0,
        hard_negative_ratio=0.3
    )
    
    valid_dataset = SimpleEdgeSparkDataset(
        valid_pkl, 
        max_points=max_points, 
        augment=False,
        negative_ratio=1.0,
        hard_negative_ratio=0.3
    )
    
    test_dataset = SimpleEdgeSparkDataset(
        test_pkl, 
        max_points=max_points, 
        augment=False,
        negative_ratio=1.0,
        hard_negative_ratio=0.3
    )
    
    # 创建数据加载器
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致
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

# 测试代码
def test_simple_dataset():
    """测试简化数据集"""
    print("测试简化数据集...")
    
    # 测试数据集
    dataset = SimpleEdgeSparkDataset(
        "dataset/train_set.pkl", 
        max_points=1000,
        augment=True,
        negative_ratio=1.0,
        hard_negative_ratio=0.3
    )
    
    # 测试样本
    sample = dataset[0]
    print(f"样本形状:")
    print(f"  源点云: {sample['source_points'].shape}")
    print(f"  目标点云: {sample['target_points'].shape}")
    print(f"  标签: {sample['label']}")
    
    # 测试数据加载器
    train_loader, valid_loader, test_loader = create_simple_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=16,
        max_points=1000,
        num_workers=0
    )
    
    print(f"\n数据加载器:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(valid_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    # 测试批次
    batch = next(iter(train_loader))
    print(f"\n批次测试:")
    print(f"  源点云: {batch['source_points'].shape}")
    print(f"  目标点云: {batch['target_points'].shape}")
    print(f"  标签: {batch['label'].shape}")
    print(f"  标签分布: {batch['label'].sum().item()}/{len(batch['label'])}")

if __name__ == "__main__":
    test_simple_dataset()