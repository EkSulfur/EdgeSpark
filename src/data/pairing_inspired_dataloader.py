"""
基于PairingNet的EdgeSpark DataLoader改进版
引入邻接矩阵、空间关系建模等关键特性
"""
import torch
import torch.utils.data as data
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from scipy.spatial.distance import cdist

class PairingInspiredDataset(data.Dataset):
    """
    基于PairingNet设计的EdgeSpark数据集
    关键改进：
    1. 邻接矩阵构建
    2. 空间关系建模
    3. 标准化的点云归一化
    4. 更好的数据预处理
    """
    
    def __init__(self, pkl_path: str, max_points: int = 2000, augment: bool = True, 
                 adjacency_k: int = 8, use_spatial_features: bool = True):
        """
        初始化数据集
        Args:
            pkl_path: pickle文件路径
            max_points: 最大点数限制
            augment: 是否使用数据增强
            adjacency_k: 邻接矩阵的k值（邻接半径）
            use_spatial_features: 是否使用空间特征
        """
        self.pkl_path = pkl_path
        self.max_points = max_points
        self.augment = augment
        self.adjacency_k = adjacency_k
        self.use_spatial_features = use_spatial_features
        
        # 加载数据
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 解析数据
        self.edge_points = self.data['full_pcd_all']  # 边缘点云列表
        self.gt_pairs = self.data['GT_pairs']  # 匹配对
        self.source_indices = self.data['source_ind']  # 源碎片匹配点索引
        self.target_indices = self.data['target_ind']  # 目标碎片匹配点索引
        
        # 计算固定的归一化参数（参考PairingNet）
        self.height_max = 1319  # PairingNet中的固定值
        
        # 创建正样本和负样本
        self.samples = self._create_samples()
        
        print(f"📊 PairingNet风格数据集加载完成: {len(self.samples)} 个样本")
        print(f"   - 正样本: {sum(1 for s in self.samples if s['label'] == 1)}")
        print(f"   - 负样本: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"   - 邻接矩阵k值: {adjacency_k}")
        print(f"   - 使用空间特征: {use_spatial_features}")
    
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
        if len(negative_pairs) > 0:
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
    
    def get_adjacency_matrix(self, points: np.ndarray) -> torch.Tensor:
        """
        构建邻接矩阵（参考PairingNet的get_adjacent2函数）
        Args:
            points: 点云数据 (n, 2)
        Returns:
            adjacency_matrix: 邻接矩阵 (max_points, max_points)
        """
        n = len(points)
        
        # 创建基础邻接矩阵
        adjacent_matrix = np.eye(n)
        temp = np.eye(n)
        
        # 添加k邻接关系
        for i in range(self.adjacency_k):
            adjacent_matrix += np.roll(temp, i + 1, axis=0)
            adjacent_matrix += np.roll(temp, -i - 1, axis=0)
        
        # 扩展到max_points大小
        full_matrix = np.zeros((self.max_points, self.max_points))
        full_matrix[:n, :n] = adjacent_matrix
        
        return torch.from_numpy(full_matrix).float()
    
    def extract_spatial_features(self, points: np.ndarray) -> np.ndarray:
        """
        提取空间特征（曲率、角度等）
        Args:
            points: 点云数据 (n, 2)
        Returns:
            spatial_features: 空间特征 (n, feature_dim)
        """
        if len(points) < 3:
            return np.zeros((len(points), 4))
        
        features = []
        
        for i in range(len(points)):
            # 获取邻近点
            prev_idx = (i - 1) % len(points)
            next_idx = (i + 1) % len(points)
            
            curr_point = points[i]
            prev_point = points[prev_idx]
            next_point = points[next_idx]
            
            # 计算向量
            vec1 = prev_point - curr_point
            vec2 = next_point - curr_point
            
            # 计算角度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(dot_product / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
            else:
                angle = 0
            
            # 计算曲率（简化版）
            if norm1 > 1e-6 and norm2 > 1e-6:
                cross_product = np.cross(vec1, vec2)
                curvature = abs(cross_product) / (norm1 * norm2)
            else:
                curvature = 0
            
            # 特征向量：[角度, 曲率, 距离1, 距离2]
            features.append([angle, curvature, norm1, norm2])
        
        return np.array(features)
    
    def normalize_points_pairing_style(self, points: np.ndarray) -> np.ndarray:
        """
        按照PairingNet风格归一化点云
        Args:
            points: 原始点云 (n, 2)
        Returns:
            normalized_points: 归一化后的点云 (n, 2)
        """
        if len(points) == 0:
            return points
        
        # PairingNet风格的归一化：/ (height_max / 2.) - 1
        normalized = points / (self.height_max / 2.0) - 1.0
        
        return normalized
    
    def pad_or_sample_points(self, points: np.ndarray) -> np.ndarray:
        """
        处理点云长度（保持有序性）
        Args:
            points: 点云数据 (n, 2)
        Returns:
            processed_points: 处理后的点云 (max_points, 2)
        """
        if len(points) >= self.max_points:
            # 等间隔采样，保持边缘点的空间连续性
            step = len(points) / self.max_points
            indices = np.round(np.arange(0, len(points), step)[:self.max_points]).astype(int)
            indices = np.clip(indices, 0, len(points) - 1)
            return points[indices]
        else:
            # 用最后一个点填充（避免破坏空间连续性）
            padding = np.tile(points[-1:], (self.max_points - len(points), 1))
            return np.vstack([points, padding])
    
    def augment_points(self, points: np.ndarray) -> np.ndarray:
        """
        数据增强：旋转、平移、噪声
        Args:
            points: 点云数据 (n, 2)
        Returns:
            augmented_points: 增强后的点云 (n, 2)
        """
        if not self.augment or len(points) == 0:
            return points
        
        # 随机旋转
        if random.random() < 0.5:
            angle = random.uniform(-np.pi/12, np.pi/12)  # ±15度
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points = points @ rotation_matrix.T
        
        # 随机平移
        if random.random() < 0.3:
            shift = np.random.uniform(-0.05, 0.05, 2)
            points = points + shift
        
        # 添加噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.005, points.shape)
            points = points + noise
        
        return points
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 获取边缘点云
        source_points = self.edge_points[sample['source_idx']].copy()
        target_points = self.edge_points[sample['target_idx']].copy()
        
        # 数据增强
        source_points = self.augment_points(source_points)
        target_points = self.augment_points(target_points)
        
        # PairingNet风格归一化
        source_points = self.normalize_points_pairing_style(source_points)
        target_points = self.normalize_points_pairing_style(target_points)
        
        # 记录原始长度
        source_length = len(source_points)
        target_length = len(target_points)
        
        # 处理点云长度
        source_points = self.pad_or_sample_points(source_points)
        target_points = self.pad_or_sample_points(target_points)
        
        # 构建邻接矩阵
        source_adj = self.get_adjacency_matrix(source_points[:source_length])
        target_adj = self.get_adjacency_matrix(target_points[:target_length])
        
        result = {
            'source_points': torch.FloatTensor(source_points),
            'target_points': torch.FloatTensor(target_points),
            'source_adj': source_adj,
            'target_adj': target_adj,
            'source_length': torch.LongTensor([source_length]),
            'target_length': torch.LongTensor([target_length]),
            'label': torch.FloatTensor([sample['label']]),
            'source_idx': sample['source_idx'],
            'target_idx': sample['target_idx']
        }
        
        # 添加空间特征（如果启用）
        if self.use_spatial_features:
            source_spatial = self.extract_spatial_features(source_points[:source_length])
            target_spatial = self.extract_spatial_features(target_points[:target_length])
            
            # 填充空间特征
            if len(source_spatial) < self.max_points:
                padding = np.zeros((self.max_points - len(source_spatial), source_spatial.shape[1]))
                source_spatial = np.vstack([source_spatial, padding])
            
            if len(target_spatial) < self.max_points:
                padding = np.zeros((self.max_points - len(target_spatial), target_spatial.shape[1]))
                target_spatial = np.vstack([target_spatial, padding])
            
            result['source_spatial'] = torch.FloatTensor(source_spatial)
            result['target_spatial'] = torch.FloatTensor(target_spatial)
        
        return result

def create_pairing_inspired_dataloaders(
    train_pkl: str, 
    valid_pkl: str, 
    test_pkl: str,
    batch_size: int = 32,
    max_points: int = 2000,
    num_workers: int = 4,
    adjacency_k: int = 8,
    use_spatial_features: bool = True
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    创建基于PairingNet的数据加载器
    """
    # 创建数据集
    train_dataset = PairingInspiredDataset(
        train_pkl, 
        max_points=max_points, 
        augment=True, 
        adjacency_k=adjacency_k,
        use_spatial_features=use_spatial_features
    )
    
    valid_dataset = PairingInspiredDataset(
        valid_pkl, 
        max_points=max_points, 
        augment=False, 
        adjacency_k=adjacency_k,
        use_spatial_features=use_spatial_features
    )
    
    test_dataset = PairingInspiredDataset(
        test_pkl, 
        max_points=max_points, 
        augment=False, 
        adjacency_k=adjacency_k,
        use_spatial_features=use_spatial_features
    )
    
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

# 测试函数
def test_pairing_inspired_dataloader():
    """测试新的数据加载器"""
    print("🔍 测试PairingNet风格的数据加载器...")
    
    try:
        # 创建数据加载器
        train_loader, valid_loader, test_loader = create_pairing_inspired_dataloaders(
            "dataset/train_set.pkl",
            "dataset/valid_set.pkl",
            "dataset/test_set.pkl",
            batch_size=4,
            max_points=1000,
            num_workers=0,
            adjacency_k=8,
            use_spatial_features=True
        )
        
        # 测试一个批次
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n📦 批次 {batch_idx + 1}:")
            print(f"   - source_points: {batch['source_points'].shape}")
            print(f"   - target_points: {batch['target_points'].shape}")
            print(f"   - source_adj: {batch['source_adj'].shape}")
            print(f"   - target_adj: {batch['target_adj'].shape}")
            print(f"   - source_length: {batch['source_length']}")
            print(f"   - target_length: {batch['target_length']}")
            print(f"   - label: {batch['label']}")
            
            if 'source_spatial' in batch:
                print(f"   - source_spatial: {batch['source_spatial'].shape}")
                print(f"   - target_spatial: {batch['target_spatial'].shape}")
            
            # 只测试第一个批次
            break
        
        print("\n✅ 数据加载器测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pairing_inspired_dataloader()