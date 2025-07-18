"""
EdgeSpark特征可视化工具
基于network_improved.py的特征提取和t-SNE可视化
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.network_improved import EdgeSparkNet
from data.improved_dataset_loader import create_dataloaders

class FeatureVisualizer:
    """EdgeSpark特征可视化器"""
    
    def __init__(self, model_path=None, device=None):
        """
        初始化特征可视化器
        Args:
            model_path: 预训练模型路径
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = EdgeSparkNet(
            segment_length=32,
            n1=160,
            n2=160,
            feature_dim=256,
            hidden_channels=64,
            temperature=1.0,
            num_samples=1
        ).to(self.device)
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 已加载预训练模型: {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ 已加载模型权重: {model_path}")
        else:
            print("⚠️  使用随机初始化的模型")
        
        self.model.eval()
        
    def extract_features(self, data_loader, max_samples=500):
        """
        Extract features
        Args:
            data_loader: Data loader
            max_samples: Maximum number of samples
        Returns:
            features: Extracted features
            labels: Corresponding labels
        """
        print("🔍 Starting feature extraction...")
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(all_features) >= max_samples:
                    break
                
                source_points = batch['source_points'].to(self.device)
                target_points = batch['target_points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 提取中间特征
                features = self._extract_intermediate_features(source_points, target_points)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Processing batch {batch_idx+1}/{len(data_loader)}")
        
        # Merge all features
        features = np.vstack(all_features)
        labels = np.vstack(all_labels).flatten()
        
        print(f"✅ Feature extraction completed: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def _extract_intermediate_features(self, source_points, target_points):
        """
        提取中间层特征
        """
        batch_size = source_points.shape[0]
        
        # 1. 段采样
        segments1_list = []
        segments2_list = []
        
        for i in range(batch_size):
            seg1 = self.model.segment_sampler(source_points[i], self.model.n1)
            seg2 = self.model.segment_sampler(target_points[i], self.model.n2)
            segments1_list.append(seg1)
            segments2_list.append(seg2)
        
        segments1 = torch.stack(segments1_list)
        segments2 = torch.stack(segments2_list)
        
        # 2. 特征编码
        features1 = self.model.segment_encoder(segments1)
        features2 = self.model.segment_encoder(segments2)
        
        # 3. 交叉注意力增强
        enhanced_features1, enhanced_features2 = self.model.cross_attention(features1, features2)
        
        # 4. 相似度计算
        similarity_matrix = self.model.similarity_computer(enhanced_features1, enhanced_features2)
        
        # 5. 特征处理
        processed_features = self.model.similarity_processor(similarity_matrix)
        attention_features = self.model.attention_pooling(similarity_matrix)
        
        # 6. 特征融合
        fused_features = processed_features + attention_features
        
        return fused_features
    
    def visualize_features(self, features, labels, save_dir="feature_visualizations", epoch=None):
        """
        可视化特征空间
        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签 (n_samples,)
            save_dir: 保存目录
            epoch: 当前epoch (可选)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 设置字体以避免中文乱码
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. t-SNE可视化
        self._plot_tsne(features_scaled, labels, save_dir, epoch)
        
        # 2. PCA可视化
        self._plot_pca(features_scaled, labels, save_dir, epoch)
        
        # 3. 特征分布分析
        self._plot_feature_distribution(features_scaled, labels, save_dir, epoch)
        
        # 4. 特征相关性分析
        self._plot_feature_correlation(features_scaled, labels, save_dir, epoch)
        
        print(f"✅ Feature visualization completed, saved to: {save_dir}")
    
    def _plot_tsne(self, features, labels, save_dir, epoch):
        """t-SNE可视化"""
        print("🎨 Generating t-SNE visualization...")
        
        # 检查样本数量
        n_samples = len(features)
        perplexity = min(30, n_samples // 4)
        
        if n_samples < 10:
            print("  ⚠️  Too few samples, skipping t-SNE visualization")
            return
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        
        # 分别绘制正负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        plt.scatter(features_2d[neg_mask, 0], features_2d[neg_mask, 1], 
                   c='red', alpha=0.6, s=50, label='Non-matching', marker='o')
        plt.scatter(features_2d[pos_mask, 0], features_2d[pos_mask, 1], 
                   c='blue', alpha=0.6, s=50, label='Matching', marker='s')
        
        plt.title(f'Feature Space t-SNE Visualization' + (f' - Epoch {epoch}' if epoch else ''))
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        pos_count = np.sum(pos_mask)
        neg_count = np.sum(neg_mask)
        plt.text(0.02, 0.98, f'Matching: {pos_count}\nNon-matching: {neg_count}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 保存
        filename = f'features_epoch_{epoch}.png' if epoch else 'features_tsne.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pca(self, features, labels, save_dir, epoch):
        """PCA可视化"""
        print("🎨 Generating PCA visualization...")
        
        # PCA降维
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        
        # 分别绘制正负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        plt.scatter(features_2d[neg_mask, 0], features_2d[neg_mask, 1], 
                   c='red', alpha=0.6, s=50, label='Non-matching', marker='o')
        plt.scatter(features_2d[pos_mask, 0], features_2d[pos_mask, 1], 
                   c='blue', alpha=0.6, s=50, label='Matching', marker='s')
        
        plt.title(f'Feature Space PCA Visualization' + (f' - Epoch {epoch}' if epoch else ''))
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存
        filename = f'features_pca_epoch_{epoch}.png' if epoch else 'features_pca.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distribution(self, features, labels, save_dir, epoch):
        """特征分布分析"""
        print("🎨 Generating feature distribution analysis...")
        
        # 计算每个特征的分离度
        separability_scores = []
        
        for i in range(features.shape[1]):
            pos_features = features[labels == 1, i]
            neg_features = features[labels == 0, i]
            
            if len(pos_features) > 0 and len(neg_features) > 0:
                # 计算两个分布的分离度
                pos_mean = np.mean(pos_features)
                neg_mean = np.mean(neg_features)
                pos_std = np.std(pos_features)
                neg_std = np.std(neg_features)
                
                # 使用Cohen's d计算分离度
                pooled_std = np.sqrt(((len(pos_features) - 1) * pos_std**2 + 
                                     (len(neg_features) - 1) * neg_std**2) / 
                                    (len(pos_features) + len(neg_features) - 2))
                
                if pooled_std > 0:
                    separability = abs(pos_mean - neg_mean) / pooled_std
                else:
                    separability = 0
                
                separability_scores.append(separability)
            else:
                separability_scores.append(0)
        
        # 绘制分离度分布
        plt.figure(figsize=(12, 8))
        plt.hist(separability_scores, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Feature Separability Distribution' + (f' - Epoch {epoch}' if epoch else ''))
        plt.xlabel('Separability Score (Cohen\'s d)')
        plt.ylabel('Number of Features')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_sep = np.mean(separability_scores)
        max_sep = np.max(separability_scores)
        plt.axvline(mean_sep, color='red', linestyle='--', label=f'Mean: {mean_sep:.3f}')
        plt.axvline(max_sep, color='green', linestyle='--', label=f'Max: {max_sep:.3f}')
        plt.legend()
        
        # 保存
        filename = f'feature_separability_epoch_{epoch}.png' if epoch else 'feature_separability.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlation(self, features, labels, save_dir, epoch):
        """特征相关性分析"""
        print("🎨 Generating feature correlation analysis...")
        
        # 计算特征相关性矩阵
        correlation_matrix = np.corrcoef(features.T)
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        
        # 只显示部分特征以避免过于密集
        max_features = min(50, features.shape[1])
        correlation_subset = correlation_matrix[:max_features, :max_features]
        
        sns.heatmap(correlation_subset, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation'})
        
        plt.title(f'Feature Correlation Matrix' + (f' - Epoch {epoch}' if epoch else ''))
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        
        # 保存
        filename = f'feature_correlation_epoch_{epoch}.png' if epoch else 'feature_correlation.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_features(self, features, labels):
        """
        Analyze feature quality
        Args:
            features: Feature matrix
            labels: Labels
        Returns:
            analysis_results: Analysis results dictionary
        """
        print("🔍 Analyzing feature quality...")
        
        pos_features = features[labels == 1]
        neg_features = features[labels == 0]
        
        # 计算基础统计
        pos_mean = np.mean(pos_features, axis=0)
        neg_mean = np.mean(neg_features, axis=0)
        pos_std = np.std(pos_features, axis=0)
        neg_std = np.std(neg_features, axis=0)
        
        # 计算类间距离
        inter_class_distance = np.linalg.norm(pos_mean - neg_mean)
        
        # 计算类内距离
        pos_intra_dist = np.mean([np.linalg.norm(f - pos_mean) for f in pos_features])
        neg_intra_dist = np.mean([np.linalg.norm(f - neg_mean) for f in neg_features])
        avg_intra_dist = (pos_intra_dist + neg_intra_dist) / 2
        
        # 可分离性比率
        separability_ratio = inter_class_distance / (avg_intra_dist + 1e-8)
        
        # 特征方差分析
        feature_variances = np.var(features, axis=0)
        active_features = np.sum(feature_variances > 1e-6)
        
        analysis_results = {
            'num_samples': len(features),
            'num_features': features.shape[1],
            'num_positive': len(pos_features),
            'num_negative': len(neg_features),
            'inter_class_distance': float(inter_class_distance),
            'avg_intra_class_distance': float(avg_intra_dist),
            'separability_ratio': float(separability_ratio),
            'active_features': int(active_features),
            'feature_variance_mean': float(np.mean(feature_variances)),
            'feature_variance_std': float(np.std(feature_variances)),
            'pos_feature_mean': float(np.mean(pos_mean)),
            'neg_feature_mean': float(np.mean(neg_mean)),
            'pos_feature_std': float(np.mean(pos_std)),
            'neg_feature_std': float(np.mean(neg_std))
        }
        
        return analysis_results

def main():
    """Main function"""
    print("🎨 EdgeSpark Feature Visualization Tool")
    print("=" * 50)
    
    # Create data loaders
    print("📚 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl", 
        "dataset/test_set.pkl",
        batch_size=32,
        max_points=1500,
        num_workers=2
    )
    
    # Find latest model
    model_path = None
    if os.path.exists("experiments"):
        exp_dirs = [d for d in os.listdir("experiments") if d.startswith("exp_")]
        if exp_dirs:
            latest_exp = sorted(exp_dirs)[-1]
            model_path = os.path.join("experiments", latest_exp, "best_model.pth")
            if not os.path.exists(model_path):
                model_path = None
    
    # Create visualizer
    visualizer = FeatureVisualizer(model_path=model_path)
    
    # Extract features
    print("🔍 Extracting features from validation set...")
    features, labels = visualizer.extract_features(val_loader, max_samples=300)
    
    # Analyze features
    analysis = visualizer.analyze_features(features, labels)
    
    print("\n📊 Feature Analysis Results:")
    print(f"  Samples: {analysis['num_samples']}")
    print(f"  Feature dimensions: {analysis['num_features']}")
    print(f"  Positive samples: {analysis['num_positive']}, Negative samples: {analysis['num_negative']}")
    print(f"  Inter-class distance: {analysis['inter_class_distance']:.4f}")
    print(f"  Intra-class distance: {analysis['avg_intra_class_distance']:.4f}")
    print(f"  Separability ratio: {analysis['separability_ratio']:.4f}")
    print(f"  Active features: {analysis['active_features']}")
    
    # Visualize features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"feature_visualizations_{timestamp}"
    visualizer.visualize_features(features, labels, save_dir)
    
    # Save analysis results
    with open(os.path.join(save_dir, "feature_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Feature visualization completed! Results saved to: {save_dir}")
    
    # Give suggestions if separability is low
    if analysis['separability_ratio'] < 1.0:
        print("\n⚠️  Low feature separability detected. Suggestions:")
        print("  1. Increase training epochs")
        print("  2. Adjust learning rate")
        print("  3. Modify network architecture")
        print("  4. Use contrastive learning")

if __name__ == "__main__":
    main()