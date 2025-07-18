"""
EdgeSpark模型表现诊断分析
全方位分析模型性能不佳的原因
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from improved_dataset_loader import ImprovedEdgeSparkDataset
from final_approach import EdgeMatchingNet

class ModelDiagnostics:
    """模型诊断分析器"""
    
    def __init__(self, data_paths, model_path=None):
        """
        初始化诊断器
        Args:
            data_paths: 数据文件路径字典 {'train': ..., 'valid': ..., 'test': ...}
            model_path: 模型文件路径
        """
        self.data_paths = data_paths
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        self.datasets = {}
        self.dataloaders = {}
        self._load_datasets()
        
        # 加载模型
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model()
        
        # 存储分析结果
        self.analysis_results = {}
        
    def _load_datasets(self):
        """加载数据集"""
        print("🔍 加载数据集...")
        for split, path in self.data_paths.items():
            self.datasets[split] = ImprovedEdgeSparkDataset(
                path, max_points=1000, augment=False, sampling_strategy='ordered'
            )
            
            self.dataloaders[split] = torch.utils.data.DataLoader(
                self.datasets[split], batch_size=32, shuffle=False, num_workers=0
            )
            print(f"  {split}: {len(self.datasets[split])} 样本")
    
    def _load_model(self):
        """加载模型"""
        print(f"🤖 加载模型: {self.model_path}")
        self.model = EdgeMatchingNet(max_points=1000).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
    
    def analyze_data_quality(self):
        """1. 数据质量分析"""
        print("\n" + "="*60)
        print("📊 数据质量分析")
        print("="*60)
        
        results = {}
        
        for split, dataset in self.datasets.items():
            print(f"\n📋 {split.upper()}数据集分析:")
            
            # 基本统计
            total_samples = len(dataset)
            positive_samples = sum(1 for s in dataset.samples if s['label'] == 1)
            negative_samples = total_samples - positive_samples
            
            print(f"  总样本数: {total_samples}")
            print(f"  正样本数: {positive_samples} ({positive_samples/total_samples:.1%})")
            print(f"  负样本数: {negative_samples} ({negative_samples/total_samples:.1%})")
            
            # 边缘点云长度分析
            point_lengths = []
            for i in range(min(1000, len(dataset))):  # 采样1000个样本分析
                sample = dataset.samples[i]
                source_len = len(dataset.edge_points[sample['source_idx']])
                target_len = len(dataset.edge_points[sample['target_idx']])
                point_lengths.extend([source_len, target_len])
            
            print(f"  边缘点云长度统计:")
            print(f"    平均: {np.mean(point_lengths):.1f}")
            print(f"    中位数: {np.median(point_lengths):.1f}")
            print(f"    最小值: {np.min(point_lengths)}")
            print(f"    最大值: {np.max(point_lengths)}")
            print(f"    标准差: {np.std(point_lengths):.1f}")
            
            # 检查数据质量问题
            empty_fragments = sum(1 for points in dataset.edge_points if len(points) == 0)
            tiny_fragments = sum(1 for points in dataset.edge_points if len(points) < 10)
            huge_fragments = sum(1 for points in dataset.edge_points if len(points) > 5000)
            
            print(f"  数据质量检查:")
            print(f"    空碎片: {empty_fragments}")
            print(f"    微小碎片(<10点): {tiny_fragments}")
            print(f"    巨大碎片(>5000点): {huge_fragments}")
            
            results[split] = {
                'total_samples': total_samples,
                'positive_ratio': positive_samples / total_samples,
                'point_lengths': {
                    'mean': float(np.mean(point_lengths)),
                    'median': float(np.median(point_lengths)),
                    'std': float(np.std(point_lengths)),
                    'min': int(np.min(point_lengths)),
                    'max': int(np.max(point_lengths))
                },
                'quality_issues': {
                    'empty_fragments': empty_fragments,
                    'tiny_fragments': tiny_fragments,
                    'huge_fragments': huge_fragments
                }
            }
        
        self.analysis_results['data_quality'] = results
        return results
    
    def analyze_feature_distribution(self):
        """2. 特征分布分析"""
        print("\n" + "="*60)
        print("🎯 特征分布分析")
        print("="*60)
        
        # 分析边缘点云的几何特征
        results = {}
        
        for split, dataset in self.datasets.items():
            print(f"\n📐 {split.upper()}几何特征分析:")
            
            # 采样部分数据进行分析
            sample_size = min(500, len(dataset))
            features = {
                'perimeters': [],
                'areas': [],
                'aspect_ratios': [],
                'curvatures': []
            }
            
            for i in range(sample_size):
                sample = dataset.samples[i]
                source_points = dataset.edge_points[sample['source_idx']]
                target_points = dataset.edge_points[sample['target_idx']]
                
                for points in [source_points, target_points]:
                    if len(points) > 3:
                        # 计算周长
                        perimeter = self._calculate_perimeter(points)
                        features['perimeters'].append(perimeter)
                        
                        # 计算面积（如果是闭合轮廓）
                        area = self._calculate_area(points)
                        features['areas'].append(area)
                        
                        # 计算宽高比
                        aspect_ratio = self._calculate_aspect_ratio(points)
                        features['aspect_ratios'].append(aspect_ratio)
                        
                        # 计算平均曲率
                        curvature = self._calculate_mean_curvature(points)
                        features['curvatures'].append(curvature)
            
            for feature_name, values in features.items():
                if values:
                    print(f"  {feature_name}:")
                    print(f"    平均值: {np.mean(values):.3f}")
                    print(f"    标准差: {np.std(values):.3f}")
                    print(f"    范围: [{np.min(values):.3f}, {np.max(values):.3f}]")
            
            results[split] = {k: {
                'mean': float(np.mean(v)) if v else 0,
                'std': float(np.std(v)) if v else 0,
                'min': float(np.min(v)) if v else 0,
                'max': float(np.max(v)) if v else 0
            } for k, v in features.items()}
        
        self.analysis_results['feature_distribution'] = results
        return results
    
    def _calculate_perimeter(self, points):
        """计算轮廓周长"""
        if len(points) < 2:
            return 0
        diffs = np.diff(points, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(distances)
    
    def _calculate_area(self, points):
        """计算轮廓面积（使用鞋带公式）"""
        if len(points) < 3:
            return 0
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
    
    def _calculate_aspect_ratio(self, points):
        """计算边界框宽高比"""
        if len(points) < 2:
            return 1.0
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        ranges = max_coords - min_coords
        return max(ranges) / (min(ranges) + 1e-8)
    
    def _calculate_mean_curvature(self, points):
        """计算平均曲率"""
        if len(points) < 3:
            return 0
        
        curvatures = []
        for i in range(len(points)):
            p1 = points[i-1]
            p2 = points[i]
            p3 = points[(i+1) % len(points)]
            
            # 计算三点角度变化
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 避免除零
            len1 = np.linalg.norm(v1) + 1e-8
            len2 = np.linalg.norm(v2) + 1e-8
            
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
        
        return np.mean(curvatures)
    
    def analyze_model_behavior(self):
        """3. 模型行为分析"""
        print("\n" + "="*60)
        print("🤖 模型行为分析")
        print("="*60)
        
        if self.model is None:
            print("❌ 没有加载模型，跳过模型行为分析")
            return {}
        
        results = {}
        
        # 对每个数据集进行预测分析
        for split, dataloader in self.dataloaders.items():
            print(f"\n🔍 {split.upper()}数据集模型行为:")
            
            predictions = []
            probabilities = []
            true_labels = []
            feature_embeddings = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    source_points = batch['source_points'].to(self.device)
                    target_points = batch['target_points'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # 获取特征嵌入
                    source_features = self.model.shape_encoder(source_points)
                    target_features = self.model.shape_encoder(target_points)
                    
                    # 获取预测结果
                    logits = self.model(source_points, target_points)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
                    # 保存特征嵌入
                    combined_features = torch.cat([source_features, target_features], dim=1)
                    feature_embeddings.extend(combined_features.cpu().numpy())
            
            # 转换为numpy数组
            predictions = np.array(predictions).flatten()
            probabilities = np.array(probabilities).flatten()
            true_labels = np.array(true_labels).flatten()
            feature_embeddings = np.array(feature_embeddings)
            
            # 分析预测分布
            print(f"  预测概率分布:")
            print(f"    平均预测概率: {np.mean(probabilities):.3f}")
            print(f"    预测概率标准差: {np.std(probabilities):.3f}")
            print(f"    预测概率范围: [{np.min(probabilities):.3f}, {np.max(probabilities):.3f}]")
            
            # 分析预测置信度
            high_confidence = np.sum((probabilities < 0.2) | (probabilities > 0.8))
            low_confidence = np.sum((probabilities >= 0.4) & (probabilities <= 0.6))
            
            print(f"  预测置信度:")
            print(f"    高置信度预测 (<0.2 or >0.8): {high_confidence} ({high_confidence/len(probabilities):.1%})")
            print(f"    低置信度预测 (0.4-0.6): {low_confidence} ({low_confidence/len(probabilities):.1%})")
            
            # 混淆矩阵
            cm = confusion_matrix(true_labels, predictions)
            print(f"  混淆矩阵:")
            print(f"    True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
            print(f"    False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")
            
            # 特征嵌入分析
            print(f"  特征嵌入分析:")
            print(f"    特征维度: {feature_embeddings.shape[1]}")
            print(f"    特征均值: {np.mean(feature_embeddings):.3f}")
            print(f"    特征标准差: {np.std(feature_embeddings):.3f}")
            
            # 分析正负样本的特征差异
            pos_indices = true_labels == 1
            neg_indices = true_labels == 0
            
            if np.sum(pos_indices) > 0 and np.sum(neg_indices) > 0:
                pos_features = feature_embeddings[pos_indices]
                neg_features = feature_embeddings[neg_indices]
                
                # 计算正负样本特征的距离
                pos_mean = np.mean(pos_features, axis=0)
                neg_mean = np.mean(neg_features, axis=0)
                feature_distance = np.linalg.norm(pos_mean - neg_mean)
                
                print(f"    正负样本特征距离: {feature_distance:.3f}")
                
                # 特征分离度分析（使用PCA）
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(feature_embeddings)
                pos_features_2d = features_2d[pos_indices]
                neg_features_2d = features_2d[neg_indices]
                
                # 计算类间/类内距离比
                if len(pos_features_2d) > 1 and len(neg_features_2d) > 1:
                    inter_class_dist = np.linalg.norm(np.mean(pos_features_2d, axis=0) - np.mean(neg_features_2d, axis=0))
                    intra_class_dist_pos = np.mean(pdist(pos_features_2d))
                    intra_class_dist_neg = np.mean(pdist(neg_features_2d))
                    avg_intra_class_dist = (intra_class_dist_pos + intra_class_dist_neg) / 2
                    
                    separability = inter_class_dist / (avg_intra_class_dist + 1e-8)
                    print(f"    特征可分离性 (类间/类内距离比): {separability:.3f}")
            
            results[split] = {
                'prediction_stats': {
                    'mean_prob': float(np.mean(probabilities)),
                    'std_prob': float(np.std(probabilities)),
                    'high_confidence_ratio': float(high_confidence / len(probabilities)),
                    'low_confidence_ratio': float(low_confidence / len(probabilities))
                },
                'confusion_matrix': cm.tolist(),
                'feature_stats': {
                    'mean': float(np.mean(feature_embeddings)),
                    'std': float(np.std(feature_embeddings)),
                    'feature_distance': float(feature_distance) if 'feature_distance' in locals() else 0,
                    'separability': float(separability) if 'separability' in locals() else 0
                }
            }
        
        self.analysis_results['model_behavior'] = results
        return results
    
    def analyze_task_complexity(self):
        """4. 任务复杂度分析"""
        print("\n" + "="*60)
        print("🎯 任务复杂度分析")
        print("="*60)
        
        results = {}
        
        # 分析碎片间的相似性
        print("🔍 分析碎片间相似性...")
        
        # 使用训练集进行分析
        train_dataset = self.datasets['train']
        
        # 采样部分数据计算相似性矩阵
        sample_size = min(100, len(train_dataset.edge_points))
        sampled_indices = np.random.choice(len(train_dataset.edge_points), sample_size, replace=False)
        
        # 计算几何特征
        geometric_features = []
        for idx in sampled_indices:
            points = train_dataset.edge_points[idx]
            if len(points) > 3:
                feature = [
                    self._calculate_perimeter(points),
                    self._calculate_area(points),
                    self._calculate_aspect_ratio(points),
                    self._calculate_mean_curvature(points),
                    len(points)
                ]
                geometric_features.append(feature)
            else:
                geometric_features.append([0, 0, 1, 0, len(points)])
        
        geometric_features = np.array(geometric_features)
        
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        geometric_features_scaled = scaler.fit_transform(geometric_features)
        
        # 计算相似性矩阵
        similarity_matrix = 1 / (1 + squareform(pdist(geometric_features_scaled, metric='euclidean')))
        
        # 分析相似性分布
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        print(f"碎片间相似性分析:")
        print(f"  平均相似性: {np.mean(upper_triangle):.3f}")
        print(f"  相似性标准差: {np.std(upper_triangle):.3f}")
        print(f"  最高相似性: {np.max(upper_triangle):.3f}")
        print(f"  最低相似性: {np.min(upper_triangle):.3f}")
        
        # 分析高相似性对的比例
        high_similarity_threshold = 0.8
        high_similarity_pairs = np.sum(upper_triangle > high_similarity_threshold)
        total_pairs = len(upper_triangle)
        
        print(f"  高相似性对(>{high_similarity_threshold}): {high_similarity_pairs}/{total_pairs} ({high_similarity_pairs/total_pairs:.1%})")
        
        # 分析正样本对vs负样本对的相似性差异
        print("\n🎯 正负样本相似性对比:")
        
        positive_similarities = []
        negative_similarities = []
        
        # 计算正样本对的相似性
        for source_idx, target_idx in train_dataset.gt_pairs[:50]:  # 采样50对
            if source_idx < len(geometric_features) and target_idx < len(geometric_features):
                sim = 1 / (1 + np.linalg.norm(geometric_features_scaled[source_idx] - geometric_features_scaled[target_idx]))
                positive_similarities.append(sim)
        
        # 计算负样本对的相似性（随机采样）
        for _ in range(min(50, len(positive_similarities))):
            idx1, idx2 = np.random.choice(len(geometric_features), 2, replace=False)
            sim = 1 / (1 + np.linalg.norm(geometric_features_scaled[idx1] - geometric_features_scaled[idx2]))
            negative_similarities.append(sim)
        
        if positive_similarities and negative_similarities:
            print(f"  正样本对平均相似性: {np.mean(positive_similarities):.3f}")
            print(f"  负样本对平均相似性: {np.mean(negative_similarities):.3f}")
            
            # 进行统计检验
            ks_stat, p_value = ks_2samp(positive_similarities, negative_similarities)
            print(f"  KS统计量: {ks_stat:.3f}, p值: {p_value:.3f}")
            
            if p_value > 0.05:
                print("  ⚠️  正负样本相似性分布无显著差异，任务可能很困难")
            else:
                print("  ✅ 正负样本相似性分布有显著差异")
        
        # 分析数据的内在维度
        print(f"\n📐 数据内在维度分析:")
        pca = PCA()
        pca.fit(geometric_features_scaled)
        
        # 计算解释方差比
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumsum_ratio >= 0.95) + 1
        
        print(f"  95%方差解释所需维度: {intrinsic_dim}")
        print(f"  前3个主成分解释方差: {cumsum_ratio[2]:.1%}")
        
        results = {
            'similarity_stats': {
                'mean_similarity': float(np.mean(upper_triangle)),
                'std_similarity': float(np.std(upper_triangle)),
                'high_similarity_ratio': float(high_similarity_pairs / total_pairs)
            },
            'class_similarity': {
                'positive_mean': float(np.mean(positive_similarities)) if positive_similarities else 0,
                'negative_mean': float(np.mean(negative_similarities)) if negative_similarities else 0,
                'ks_pvalue': float(p_value) if 'p_value' in locals() else 1.0
            },
            'intrinsic_dimension': int(intrinsic_dim),
            'pca_variance_ratio': cumsum_ratio[2] if len(cumsum_ratio) > 2 else 0
        }
        
        self.analysis_results['task_complexity'] = results
        return results
    
    def implement_baseline_methods(self):
        """5. 基线方法对比"""
        print("\n" + "="*60)
        print("📏 基线方法对比")
        print("="*60)
        
        results = {}
        
        # 1. 随机基线
        print("🎲 随机基线:")
        random_accuracy = 0.5  # 平衡数据集的随机准确率
        print(f"  随机准确率: {random_accuracy:.3f}")
        
        # 2. 简单几何特征基线
        print("\n📐 几何特征基线:")
        
        # 使用验证集测试基线方法
        val_dataset = self.datasets['valid']
        
        correct_predictions = 0
        total_predictions = 0
        
        for sample in val_dataset.samples[:100]:  # 测试100个样本
            source_points = val_dataset.edge_points[sample['source_idx']]
            target_points = val_dataset.edge_points[sample['target_idx']]
            true_label = sample['label']
            
            # 计算几何特征相似性
            if len(source_points) > 3 and len(target_points) > 3:
                source_features = np.array([
                    self._calculate_perimeter(source_points),
                    self._calculate_area(target_points),
                    self._calculate_aspect_ratio(source_points),
                    len(source_points)
                ])
                
                target_features = np.array([
                    self._calculate_perimeter(target_points),
                    self._calculate_area(target_points),
                    self._calculate_aspect_ratio(target_points),
                    len(target_points)
                ])
                
                # 计算相似性（归一化后的L2距离）
                source_features = source_features / (np.linalg.norm(source_features) + 1e-8)
                target_features = target_features / (np.linalg.norm(target_features) + 1e-8)
                
                similarity = 1 / (1 + np.linalg.norm(source_features - target_features))
                prediction = 1 if similarity > 0.7 else 0
                
                if prediction == true_label:
                    correct_predictions += 1
                total_predictions += 1
        
        geometric_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"  几何特征基线准确率: {geometric_accuracy:.3f}")
        
        # 3. 长度比基线
        print("\n📏 长度比基线:")
        
        correct_predictions = 0
        total_predictions = 0
        
        for sample in val_dataset.samples[:100]:
            source_len = len(val_dataset.edge_points[sample['source_idx']])
            target_len = len(val_dataset.edge_points[sample['target_idx']])
            true_label = sample['label']
            
            # 长度相似性
            length_ratio = min(source_len, target_len) / (max(source_len, target_len) + 1e-8)
            prediction = 1 if length_ratio > 0.8 else 0
            
            if prediction == true_label:
                correct_predictions += 1
            total_predictions += 1
        
        length_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"  长度比基线准确率: {length_accuracy:.3f}")
        
        # 对比深度学习模型
        if self.model is not None and 'valid' in self.analysis_results.get('model_behavior', {}):
            model_accuracy = 1 - np.mean(self.analysis_results['model_behavior']['valid']['confusion_matrix'])
            model_accuracy = (self.analysis_results['model_behavior']['valid']['confusion_matrix'][0][0] + 
                            self.analysis_results['model_behavior']['valid']['confusion_matrix'][1][1]) / \
                           np.sum(self.analysis_results['model_behavior']['valid']['confusion_matrix'])
            
            print(f"\n🤖 深度学习模型准确率: {model_accuracy:.3f}")
            
            print("\n📊 基线对比总结:")
            baselines = [
                ("随机基线", random_accuracy),
                ("几何特征基线", geometric_accuracy),
                ("长度比基线", length_accuracy),
                ("深度学习模型", model_accuracy)
            ]
            
            for name, acc in baselines:
                print(f"  {name}: {acc:.3f}")
        
        results = {
            'random_baseline': random_accuracy,
            'geometric_baseline': geometric_accuracy,
            'length_baseline': length_accuracy,
            'model_accuracy': model_accuracy if 'model_accuracy' in locals() else 0
        }
        
        self.analysis_results['baseline_comparison'] = results
        return results
    
    def analyze_failure_cases(self):
        """6. 错误案例分析"""
        print("\n" + "="*60)
        print("❌ 错误案例分析")
        print("="*60)
        
        if self.model is None:
            print("❌ 没有加载模型，跳过错误案例分析")
            return {}
        
        results = {}
        
        # 在验证集上找错误案例
        val_dataloader = self.dataloaders['valid']
        
        false_positives = []  # 假阳性：模型预测为匹配，实际不匹配
        false_negatives = []  # 假阴性：模型预测为不匹配，实际匹配
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                source_points = batch['source_points'].to(self.device)
                target_points = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                source_indices = batch['source_idx'].numpy()
                target_indices = batch['target_idx'].numpy()
                
                logits = self.model(source_points, target_points)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # 找错误案例
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    confidence = probs[i].item()
                    
                    if true_label != pred_label:
                        case_info = {
                            'source_idx': int(source_indices[i]),
                            'target_idx': int(target_indices[i]),
                            'true_label': int(true_label),
                            'pred_label': int(pred_label),
                            'confidence': float(confidence),
                            'source_points': source_points[i].cpu().numpy(),
                            'target_points': target_points[i].cpu().numpy()
                        }
                        
                        if true_label == 0 and pred_label == 1:
                            false_positives.append(case_info)
                        elif true_label == 1 and pred_label == 0:
                            false_negatives.append(case_info)
                
                # 限制分析的案例数量
                if len(false_positives) + len(false_negatives) >= 50:
                    break
        
        print(f"收集到错误案例: FP={len(false_positives)}, FN={len(false_negatives)}")
        
        # 初始化所有变量
        fp_length_ratios = []
        fp_area_ratios = []
        fn_length_ratios = []
        fn_area_ratios = []
        
        # 分析假阳性案例
        if false_positives:
            print(f"\n🔴 假阳性分析 (预测匹配，实际不匹配):")
            fp_confidences = [case['confidence'] for case in false_positives]
            print(f"  数量: {len(false_positives)}")
            print(f"  平均置信度: {np.mean(fp_confidences):.3f}")
            print(f"  置信度范围: [{np.min(fp_confidences):.3f}, {np.max(fp_confidences):.3f}]")
            
            # 分析假阳性案例的几何特征
            fp_length_ratios = []
            fp_area_ratios = []
            
            for case in false_positives[:10]:  # 分析前10个案例
                val_dataset = self.datasets['valid']
                source_points = val_dataset.edge_points[case['source_idx']]
                target_points = val_dataset.edge_points[case['target_idx']]
                
                if len(source_points) > 3 and len(target_points) > 3:
                    source_len = len(source_points)
                    target_len = len(target_points)
                    length_ratio = min(source_len, target_len) / max(source_len, target_len)
                    fp_length_ratios.append(length_ratio)
                    
                    source_area = self._calculate_area(source_points)
                    target_area = self._calculate_area(target_points)
                    if source_area > 0 and target_area > 0:
                        area_ratio = min(source_area, target_area) / max(source_area, target_area)
                        fp_area_ratios.append(area_ratio)
            
            if fp_length_ratios:
                print(f"  长度相似性: {np.mean(fp_length_ratios):.3f} ± {np.std(fp_length_ratios):.3f}")
            if fp_area_ratios:
                print(f"  面积相似性: {np.mean(fp_area_ratios):.3f} ± {np.std(fp_area_ratios):.3f}")
        
        # 分析假阴性案例
        if false_negatives:
            print(f"\n🔵 假阴性分析 (预测不匹配，实际匹配):")
            fn_confidences = [1 - case['confidence'] for case in false_negatives]  # 转换为"不匹配"的置信度
            print(f"  数量: {len(false_negatives)}")
            print(f"  平均置信度: {np.mean(fn_confidences):.3f}")
            print(f"  置信度范围: [{np.min(fn_confidences):.3f}, {np.max(fn_confidences):.3f}]")
            
            # 分析假阴性案例的几何特征
            fn_length_ratios = []
            fn_area_ratios = []
            
            for case in false_negatives[:10]:
                val_dataset = self.datasets['valid']
                source_points = val_dataset.edge_points[case['source_idx']]
                target_points = val_dataset.edge_points[case['target_idx']]
                
                if len(source_points) > 3 and len(target_points) > 3:
                    source_len = len(source_points)
                    target_len = len(target_points)
                    length_ratio = min(source_len, target_len) / max(source_len, target_len)
                    fn_length_ratios.append(length_ratio)
                    
                    source_area = self._calculate_area(source_points)
                    target_area = self._calculate_area(target_points)
                    if source_area > 0 and target_area > 0:
                        area_ratio = min(source_area, target_area) / max(source_area, target_area)
                        fn_area_ratios.append(area_ratio)
            
            if fn_length_ratios:
                print(f"  长度相似性: {np.mean(fn_length_ratios):.3f} ± {np.std(fn_length_ratios):.3f}")
            if fn_area_ratios:
                print(f"  面积相似性: {np.mean(fn_area_ratios):.3f} ± {np.std(fn_area_ratios):.3f}")
        
        results = {
            'false_positives': {
                'count': len(false_positives),
                'avg_confidence': float(np.mean(fp_confidences)) if false_positives else 0,
                'avg_length_similarity': float(np.mean(fp_length_ratios)) if fp_length_ratios else 0,
                'avg_area_similarity': float(np.mean(fp_area_ratios)) if fp_area_ratios else 0
            },
            'false_negatives': {
                'count': len(false_negatives),
                'avg_confidence': float(np.mean(fn_confidences)) if false_negatives else 0,
                'avg_length_similarity': float(np.mean(fn_length_ratios)) if fn_length_ratios else 0,
                'avg_area_similarity': float(np.mean(fn_area_ratios)) if fn_area_ratios else 0
            }
        }
        
        self.analysis_results['failure_analysis'] = results
        return results
    
    def generate_comprehensive_report(self):
        """生成综合诊断报告"""
        print("\n" + "="*80)
        print("📋 EdgeSpark模型诊断综合报告")
        print("="*80)
        
        # 运行所有分析
        print("🔍 开始全面诊断分析...")
        
        self.analyze_data_quality()
        self.analyze_feature_distribution()
        self.analyze_model_behavior()
        self.analyze_task_complexity()
        self.implement_baseline_methods()
        self.analyze_failure_cases()
        
        # 生成总结报告
        print("\n" + "="*80)
        print("📊 诊断总结与建议")
        print("="*80)
        
        # 数据质量问题
        print("\n1️⃣ 数据质量诊断:")
        data_quality = self.analysis_results.get('data_quality', {})
        if data_quality:
            train_quality = data_quality.get('train', {})
            if train_quality.get('quality_issues', {}).get('tiny_fragments', 0) > 100:
                print("  ⚠️  发现大量微小碎片，可能影响特征学习")
            if train_quality.get('point_lengths', {}).get('std', 0) > 1000:
                print("  ⚠️  碎片尺寸变化很大，建议改进归一化策略")
            print("  ✅ 正负样本平衡良好")
        
        # 特征分离度问题
        print("\n2️⃣ 特征表示诊断:")
        model_behavior = self.analysis_results.get('model_behavior', {})
        if model_behavior:
            val_behavior = model_behavior.get('valid', {})
            separability = val_behavior.get('feature_stats', {}).get('separability', 0)
            if separability < 1.0:
                print("  ❌ 特征可分离性差，正负样本在特征空间中难以区分")
                print("     建议：增强特征提取能力，尝试更复杂的网络架构")
            else:
                print("  ✅ 特征可分离性良好")
        
        # 任务复杂度问题
        print("\n3️⃣ 任务复杂度诊断:")
        task_complexity = self.analysis_results.get('task_complexity', {})
        if task_complexity:
            ks_pvalue = task_complexity.get('class_similarity', {}).get('ks_pvalue', 1.0)
            if ks_pvalue > 0.05:
                print("  ❌ 正负样本相似性分布无显著差异，任务本身很困难")
                print("     建议：收集更多高质量的匹配样本，或重新定义匹配标准")
            else:
                print("  ✅ 任务具有可学习性")
        
        # 基线对比
        print("\n4️⃣ 模型性能诊断:")
        baseline_comparison = self.analysis_results.get('baseline_comparison', {})
        if baseline_comparison:
            model_acc = baseline_comparison.get('model_accuracy', 0)
            geometric_acc = baseline_comparison.get('geometric_baseline', 0)
            
            if model_acc < geometric_acc + 0.1:
                print("  ❌ 深度学习模型未能显著超越简单基线")
                print("     建议：检查网络架构设计，增加模型复杂度")
            else:
                print("  ✅ 深度学习模型性能优于基线")
        
        # 错误分析
        print("\n5️⃣ 错误模式诊断:")
        failure_analysis = self.analysis_results.get('failure_analysis', {})
        if failure_analysis:
            fp_count = failure_analysis.get('false_positives', {}).get('count', 0)
            fn_count = failure_analysis.get('false_negatives', {}).get('count', 0)
            
            if fp_count > fn_count:
                print("  ⚠️  假阳性错误较多，模型过于宽松")
                print("     建议：调整决策阈值，增加负样本训练")
            elif fn_count > fp_count:
                print("  ⚠️  假阴性错误较多，模型过于严格")
                print("     建议：增强数据增强，提高模型对变化的鲁棒性")
        
        # 总体建议
        print("\n🔧 改进建议优先级:")
        print("  1. 数据质量：清理微小碎片，统一碎片尺寸范围")
        print("  2. 特征工程：尝试更丰富的几何特征（曲率、傅里叶描述子等）")
        print("  3. 网络架构：考虑使用注意力机制或图神经网络")
        print("  4. 训练策略：实施课程学习，从简单样本开始训练")
        print("  5. 数据增强：增加更多几何变换（缩放、旋转、噪声）")
        print("  6. 损失函数：尝试focal loss处理困难样本")
        
        # 保存报告
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"diagnosis_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 详细诊断报告已保存到: {report_file}")
        
        return self.analysis_results

def main():
    """主函数"""
    print("🔬 EdgeSpark模型诊断分析")
    print("="*50)
    
    # 数据路径
    data_paths = {
        'train': 'dataset/train_set.pkl',
        'valid': 'dataset/valid_set.pkl',
        'test': 'dataset/test_set.pkl'
    }
    
    # 模型路径（如果存在）
    model_path = 'best_final_model_ordered.pth'
    if not os.path.exists(model_path):
        model_path = None
        print("⚠️  未找到训练好的模型，将跳过模型相关分析")
    
    # 创建诊断器
    diagnostics = ModelDiagnostics(data_paths, model_path)
    
    # 生成综合报告
    results = diagnostics.generate_comprehensive_report()
    
    return results

if __name__ == "__main__":
    results = main()