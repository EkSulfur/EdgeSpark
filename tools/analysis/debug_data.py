"""
数据调试脚本
分析数据质量和分布
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

def analyze_dataset(pkl_path):
    """分析数据集"""
    print(f"\n=== 分析数据集: {pkl_path} ===")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    edge_points = data['full_pcd_all']
    gt_pairs = data['GT_pairs']
    
    print(f"边缘点云数量: {len(edge_points)}")
    print(f"匹配对数量: {len(gt_pairs)}")
    
    # 分析点云统计
    point_counts = [len(points) for points in edge_points]
    print(f"点数统计:")
    print(f"  最小: {min(point_counts)}")
    print(f"  最大: {max(point_counts)}")
    print(f"  平均: {np.mean(point_counts):.1f}")
    print(f"  中位数: {np.median(point_counts):.1f}")
    
    # 检查空点云
    empty_count = sum(1 for points in edge_points if len(points) == 0)
    print(f"空点云数量: {empty_count}")
    
    # 分析边缘点云的空间分布
    all_points = []
    for points in edge_points:
        if len(points) > 0:
            all_points.extend(points)
    
    if all_points:
        all_points = np.array(all_points)
        print(f"全局点云统计:")
        print(f"  X范围: [{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}]")
        print(f"  Y范围: [{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}]")
        print(f"  X标准差: {all_points[:, 0].std():.3f}")
        print(f"  Y标准差: {all_points[:, 1].std():.3f}")
    
    # 分析匹配对
    print(f"\n匹配对分析:")
    source_indices = [pair[0] for pair in gt_pairs]
    target_indices = [pair[1] for pair in gt_pairs]
    
    print(f"源索引范围: {min(source_indices)} - {max(source_indices)}")
    print(f"目标索引范围: {min(target_indices)} - {max(target_indices)}")
    
    # 检查是否有重复的匹配对
    unique_pairs = set(tuple(pair) for pair in gt_pairs)
    print(f"唯一匹配对: {len(unique_pairs)}")
    print(f"重复匹配对: {len(gt_pairs) - len(unique_pairs)}")
    
    return edge_points, gt_pairs

def compute_simple_features(edge_points):
    """计算简单的几何特征"""
    features = []
    
    for points in edge_points:
        if len(points) == 0:
            features.append(np.zeros(8))
            continue
        
        points = np.array(points)
        
        # 几何特征
        centroid = np.mean(points, axis=0)
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        
        # 统计特征
        std = np.std(points, axis=0)
        
        feature = np.concatenate([centroid, bbox_min, bbox_max, bbox_size, std])
        features.append(feature)
    
    return np.array(features)

def analyze_feature_separability(edge_points, gt_pairs):
    """分析特征的可分性"""
    print(f"\n=== 特征可分性分析 ===")
    
    # 计算特征
    features = compute_simple_features(edge_points)
    print(f"特征维度: {features.shape}")
    
    # 分析正样本对的特征相似性
    positive_distances = []
    for pair in gt_pairs:
        source_idx, target_idx = pair[0], pair[1]
        if source_idx < len(features) and target_idx < len(features):
            feat1 = features[source_idx]
            feat2 = features[target_idx]
            dist = np.linalg.norm(feat1 - feat2)
            positive_distances.append(dist)
    
    # 分析负样本对的特征相似性
    negative_distances = []
    np.random.seed(42)
    for _ in range(len(gt_pairs)):
        idx1, idx2 = np.random.choice(len(features), 2, replace=False)
        pair_exists = any((p[0] == idx1 and p[1] == idx2) or (p[0] == idx2 and p[1] == idx1) for p in gt_pairs)
        if not pair_exists:
            feat1 = features[idx1]
            feat2 = features[idx2]
            dist = np.linalg.norm(feat1 - feat2)
            negative_distances.append(dist)
    
    print(f"正样本特征距离统计:")
    print(f"  平均: {np.mean(positive_distances):.4f}")
    print(f"  标准差: {np.std(positive_distances):.4f}")
    print(f"  范围: [{np.min(positive_distances):.4f}, {np.max(positive_distances):.4f}]")
    
    print(f"负样本特征距离统计:")
    print(f"  平均: {np.mean(negative_distances):.4f}")
    print(f"  标准差: {np.std(negative_distances):.4f}")
    print(f"  范围: [{np.min(negative_distances):.4f}, {np.max(negative_distances):.4f}]")
    
    # 可分性分析
    pos_mean = np.mean(positive_distances)
    neg_mean = np.mean(negative_distances)
    
    print(f"\n可分性分析:")
    print(f"  正样本距离更小: {pos_mean < neg_mean}")
    print(f"  距离差: {abs(pos_mean - neg_mean):.4f}")
    
    # 重叠分析
    pos_95 = np.percentile(positive_distances, 95)
    neg_5 = np.percentile(negative_distances, 5)
    
    print(f"  正样本95%分位数: {pos_95:.4f}")
    print(f"  负样本5%分位数: {neg_5:.4f}")
    print(f"  特征重叠: {pos_95 > neg_5}")
    
    return positive_distances, negative_distances

def test_simple_classifier(edge_points, gt_pairs):
    """测试简单分类器"""
    print(f"\n=== 简单分类器测试 ===")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    
    # 准备数据
    features = compute_simple_features(edge_points)
    
    # 创建样本
    X, y = [], []
    
    # 正样本
    for pair in gt_pairs:
        source_idx, target_idx = pair[0], pair[1]
        if source_idx < len(features) and target_idx < len(features):
            feat1 = features[source_idx]
            feat2 = features[target_idx]
            combined_feat = np.concatenate([feat1, feat2, feat1 - feat2, feat1 * feat2])
            X.append(combined_feat)
            y.append(1)
    
    # 负样本
    np.random.seed(42)
    for _ in range(len(gt_pairs)):
        idx1, idx2 = np.random.choice(len(features), 2, replace=False)
        pair_exists = any((p[0] == idx1 and p[1] == idx2) or (p[0] == idx2 and p[1] == idx1) for p in gt_pairs)
        if not pair_exists:
            feat1 = features[idx1]
            feat2 = features[idx2]
            combined_feat = np.concatenate([feat1, feat2, feat1 - feat2, feat1 * feat2])
            X.append(combined_feat)
            y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"样本数量: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"正样本比例: {np.mean(y):.3f}")
    
    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 评估
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"随机森林准确率: {acc:.4f}")
    
    if acc > 0.6:
        print("✅ 特征具有一定的区分能力")
    else:
        print("❌ 特征区分能力较差")
    
    return acc

def main():
    """主函数"""
    print("🔍 EdgeSpark数据调试分析")
    print("=" * 60)
    
    # 分析训练集
    train_edge_points, train_gt_pairs = analyze_dataset("dataset/train_set.pkl")
    
    # 分析验证集
    val_edge_points, val_gt_pairs = analyze_dataset("dataset/valid_set.pkl")
    
    # 特征可分性分析
    pos_dist, neg_dist = analyze_feature_separability(train_edge_points, train_gt_pairs)
    
    # 简单分类器测试
    acc = test_simple_classifier(train_edge_points, train_gt_pairs)
    
    print(f"\n=== 总结 ===")
    if acc > 0.7:
        print("🎉 数据质量良好，深度学习模型应该能够学习")
        print("💡 建议：检查网络架构和训练策略")
    elif acc > 0.6:
        print("⚠️  数据质量一般，可能需要更复杂的特征")
        print("💡 建议：尝试更深的网络或更好的特征工程")
    else:
        print("🔴 数据质量较差，基本几何特征无法区分")
        print("💡 建议：检查数据质量或寻找更好的特征表示")

if __name__ == "__main__":
    main()