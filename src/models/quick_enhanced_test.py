"""
快速测试增强特征提取器的效果
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from improved_dataset_loader import create_improved_dataloaders
from enhanced_feature_extractor import EnhancedFragmentMatcher, MultiScaleFeatureExtractor

def test_feature_separability():
    """测试特征可分离性"""
    print("🔍 快速测试增强特征提取器")
    print("=" * 50)
    
    # 创建数据加载器（小batch）
    print("📚 创建数据加载器...")
    train_loader, val_loader, test_loader = create_improved_dataloaders(
        "dataset/train_set.pkl",
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=32,
        max_points=1000,
        num_workers=0,
        sampling_strategy='ordered'
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 测试增强特征提取器
    enhanced_model = EnhancedFragmentMatcher(max_points=1000).to(device)
    
    # 测试原始特征提取器（用于对比）
    simple_extractor = MultiScaleFeatureExtractor(max_points=1000).to(device)
    
    print(f"📊 增强模型参数: {sum(p.numel() for p in enhanced_model.parameters()):,}")
    
    # 从验证集提取特征
    enhanced_features1 = []
    enhanced_features2 = []
    simple_features1 = []
    simple_features2 = []
    labels = []
    
    enhanced_model.eval()
    simple_extractor.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # 只测试前5个batch
                break
                
            points1 = batch['source_points'].to(device)
            points2 = batch['target_points'].to(device)
            batch_labels = batch['label'].to(device)
            
            # 增强特征提取
            _, feat1_enhanced, feat2_enhanced = enhanced_model(points1, points2)
            
            # 简单特征提取
            feat1_simple = simple_extractor(points1)
            feat2_simple = simple_extractor(points2)
            
            enhanced_features1.extend(feat1_enhanced.cpu().numpy())
            enhanced_features2.extend(feat2_enhanced.cpu().numpy())
            simple_features1.extend(feat1_simple.cpu().numpy())
            simple_features2.extend(feat2_simple.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
            print(f"  处理了 {i+1}/5 个batch")
    
    # 转换为numpy
    enhanced_features1 = np.array(enhanced_features1)
    enhanced_features2 = np.array(enhanced_features2)
    simple_features1 = np.array(simple_features1)
    simple_features2 = np.array(simple_features2)
    labels = np.array(labels).flatten()
    
    print(f"📈 提取了 {len(labels)} 个样本的特征")
    print(f"   正样本: {np.sum(labels == 1)}")
    print(f"   负样本: {np.sum(labels == 0)}")
    
    # 计算特征可分离性
    def compute_separability(feat1, feat2, labels):
        """计算特征可分离性"""
        combined = np.concatenate([feat1, feat2], axis=1)
        
        pos_features = combined[labels == 1]
        neg_features = combined[labels == 0]
        
        if len(pos_features) < 2 or len(neg_features) < 2:
            return 0.0
        
        # 类间距离
        pos_center = np.mean(pos_features, axis=0)
        neg_center = np.mean(neg_features, axis=0)
        inter_dist = np.linalg.norm(pos_center - neg_center)
        
        # 类内距离
        pos_distances = [np.linalg.norm(f - pos_center) for f in pos_features]
        neg_distances = [np.linalg.norm(f - neg_center) for f in neg_features]
        intra_dist = (np.mean(pos_distances) + np.mean(neg_distances)) / 2
        
        return inter_dist / (intra_dist + 1e-8)
    
    # 计算可分离性
    enhanced_separability = compute_separability(enhanced_features1, enhanced_features2, labels)
    simple_separability = compute_separability(simple_features1, simple_features2, labels)
    
    print(f"\n📊 特征可分离性对比:")
    print(f"   增强特征提取器: {enhanced_separability:.4f}")
    print(f"   简单特征提取器: {simple_separability:.4f}")
    print(f"   改进倍数: {enhanced_separability / (simple_separability + 1e-8):.2f}x")
    
    # 可视化特征空间
    def visualize_features(feat1, feat2, labels, title, save_name):
        """可视化特征空间"""
        combined = np.concatenate([feat1, feat2], axis=1)
        
        # 使用PCA降维（更快）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(combined)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if label == 0 else 'blue' for label in labels]
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
        plt.title(f'{title} (可分离性: {compute_separability(feat1, feat2, labels):.4f})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(['Non-matching', 'Matching'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   保存可视化图: {save_name}.png")
    
    print(f"\n📈 生成特征空间可视化...")
    visualize_features(enhanced_features1, enhanced_features2, labels, 
                      "增强特征提取器", "enhanced_features")
    visualize_features(simple_features1, simple_features2, labels, 
                      "简单特征提取器", "simple_features")
    
    # 测试各个几何特征的贡献
    print(f"\n🔍 分析几何特征贡献...")
    
    from enhanced_feature_extractor import GeometricFeatureExtractor
    geo_extractor = GeometricFeatureExtractor()
    
    # 从数据集提取几何特征
    dataset = val_loader.dataset
    geometric_separabilities = {}
    
    for feature_type in ['curvature', 'angle', 'fourier', 'moment', 'distance']:
        type_features1 = []
        type_features2 = []
        
        for i in range(min(100, len(dataset))):  # 测试100个样本
            sample = dataset.samples[i]
            source_points = dataset.edge_points[sample['source_idx']]
            target_points = dataset.edge_points[sample['target_idx']]
            
            # 去除padding
            source_valid = source_points[~((source_points == -999).all(axis=1))]
            target_valid = target_points[~((target_points == -999).all(axis=1))]
            
            if len(source_valid) > 3 and len(target_valid) > 3:
                source_geo = geo_extractor.extract_all_features(source_valid)
                target_geo = geo_extractor.extract_all_features(target_valid)
                
                type_features1.append(source_geo[feature_type])
                type_features2.append(target_geo[feature_type])
        
        if len(type_features1) > 10:
            type_features1 = np.array(type_features1)
            type_features2 = np.array(type_features2)
            type_labels = np.array([dataset.samples[i]['label'] for i in range(len(type_features1))])
            
            separability = compute_separability(type_features1, type_features2, type_labels)
            geometric_separabilities[feature_type] = separability
            
            print(f"   {feature_type:>8}: {separability:.4f}")
    
    # 结果总结
    print(f"\n" + "=" * 50)
    print("📊 总结")
    print("=" * 50)
    
    if enhanced_separability > simple_separability * 1.5:
        print("🎉 增强特征提取器显著改善了特征可分离性！")
        print("💡 几何特征工程是有效的")
    elif enhanced_separability > simple_separability * 1.1:
        print("✅ 增强特征提取器有一定改善")
        print("💡 还有进一步优化空间")
    else:
        print("⚠️  增强特征提取器改善有限")
        print("💡 需要重新思考特征工程策略")
    
    # 找到最有效的几何特征
    if geometric_separabilities:
        best_geo_feature = max(geometric_separabilities.items(), key=lambda x: x[1])
        print(f"🏆 最有效的几何特征: {best_geo_feature[0]} (可分离性: {best_geo_feature[1]:.4f})")
    
    return {
        'enhanced_separability': enhanced_separability,
        'simple_separability': simple_separability,
        'geometric_separabilities': geometric_separabilities,
        'improvement_ratio': enhanced_separability / (simple_separability + 1e-8)
    }

if __name__ == "__main__":
    results = test_feature_separability()
    print(f"\n🎯 最终结果: {results}")