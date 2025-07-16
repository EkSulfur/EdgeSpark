import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(pkl_path):
    """深入分析数据集的结构和特征"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"=== 分析数据集: {pkl_path} ===\n")
    
    # 1. 基本信息
    print("1. 基本信息:")
    print(f"   总碎片数: {len(data['full_pcd_all'])}")
    print(f"   匹配对数: {len(data['GT_pairs'])}")
    print()
    
    # 2. 边缘点云数据分析
    print("2. 边缘点云数据 (full_pcd_all):")
    edge_points = data['full_pcd_all']
    edge_lengths = [len(pcd) for pcd in edge_points]
    
    print(f"   边缘点数量统计:")
    print(f"   - 平均点数: {np.mean(edge_lengths):.2f}")
    print(f"   - 最小点数: {min(edge_lengths)}")
    print(f"   - 最大点数: {max(edge_lengths)}")
    print(f"   - 点数分布: {np.percentile(edge_lengths, [25, 50, 75])}")
    print()
    
    # 3. 形状信息分析
    print("3. 形状信息 (shape_all):")
    shapes = data['shape_all']
    print(f"   形状数据示例: {shapes[:5]}")
    print()
    
    # 4. 匹配对分析
    print("4. 匹配对 (GT_pairs):")
    gt_pairs = data['GT_pairs']
    print(f"   匹配对示例: {gt_pairs[:5]}")
    print(f"   匹配对格式: 每对包含 {len(gt_pairs[0])} 个元素")
    print()
    
    # 5. 匹配点索引分析
    print("5. 匹配点索引:")
    source_ind = data['source_ind']
    target_ind = data['target_ind']
    
    source_lengths = [len(ind) for ind in source_ind]
    target_lengths = [len(ind) for ind in target_ind]
    
    print(f"   源碎片匹配点数:")
    print(f"   - 平均点数: {np.mean(source_lengths):.2f}")
    print(f"   - 点数范围: [{min(source_lengths)}, {max(source_lengths)}]")
    print(f"   目标碎片匹配点数:")
    print(f"   - 平均点数: {np.mean(target_lengths):.2f}")
    print(f"   - 点数范围: [{min(target_lengths)}, {max(target_lengths)}]")
    print()
    
    # 6. 数据示例
    print("6. 数据示例:")
    print(f"   第一个碎片边缘点云形状: {edge_points[0].shape}")
    print(f"   第一个碎片边缘点云前5个点:")
    print(f"   {edge_points[0][:5]}")
    print()
    
    if len(gt_pairs) > 0:
        pair_idx = 0
        pair = gt_pairs[pair_idx]
        print(f"   第一个匹配对: {pair}")
        print(f"   源碎片索引: {source_ind[pair_idx][:10]}...")
        print(f"   目标碎片索引: {target_ind[pair_idx][:10]}...")
        print()
    
    return data

def visualize_fragments(data, num_fragments=5):
    """可视化几个碎片的边缘点云"""
    edge_points = data['full_pcd_all']
    
    fig, axes = plt.subplots(1, num_fragments, figsize=(15, 3))
    if num_fragments == 1:
        axes = [axes]
    
    for i in range(num_fragments):
        pcd = edge_points[i]
        axes[i].scatter(pcd[:, 0], pcd[:, 1], s=1, alpha=0.7)
        axes[i].set_title(f'碎片 {i}')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fragment_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("碎片可视化保存为 fragment_visualization.png")

if __name__ == "__main__":
    # 分析训练集
    train_data = analyze_dataset("dataset/train_set.pkl")
    
    # 可视化几个碎片
    visualize_fragments(train_data, num_fragments=5)