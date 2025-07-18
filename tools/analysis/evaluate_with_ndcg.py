"""
NDCG评估脚本：使用PairingNet的NDCG方法评估EdgeSpark final_approach模型
基于PairingNet/PairingNet Code/utils/NDCG.py的评估方法
"""
import torch
import numpy as np
import pickle
import os
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# 添加路径以导入EdgeSpark模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'PairingNet', 'PairingNet Code', 'utils'))

from final_approach import EdgeMatchingNet
from improved_dataset_loader import create_improved_dataloaders
from NDCG import calculate_NDCG, ndcg

class EdgeSparkNDCGEvaluator:
    """
    EdgeSpark NDCG评估器
    使用PairingNet的NDCG评估方法来评估final_approach模型
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化评估器
        
        Args:
            model_path: 训练好的模型文件路径
            device: 计算设备 (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        
        print(f"🔧 使用设备: {self.device}")
        
    def load_model(self):
        """加载训练好的模型"""
        print(f"📥 加载模型: {self.model_path}")
        
        # 创建模型实例
        self.model = EdgeMatchingNet(max_points=1000).to(self.device)
        
        # 加载权重
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
    
    def load_dataset(self, dataset_path):
        """
        加载数据集
        
        Args:
            dataset_path: 数据集pkl文件路径
            
        Returns:
            tuple: (fragments, gt_pairs)
        """
        print(f"📚 加载数据集: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        fragments = data['full_pcd_all']  # 边缘点云数据
        gt_pairs = data['GT_pairs']       # 真实匹配对
        
        print(f"✅ 数据集加载完成")
        print(f"   - 片段数量: {len(fragments)}")
        print(f"   - 真实匹配对: {len(gt_pairs)}")
        
        return fragments, gt_pairs
    
    def preprocess_fragment(self, fragment, max_points=1000):
        """
        预处理片段数据
        
        Args:
            fragment: 片段边缘点云 (numpy array)
            max_points: 最大点数
            
        Returns:
            torch.Tensor: 预处理后的点云 (1, max_points, 2)
        """
        # 确保是numpy数组
        if not isinstance(fragment, np.ndarray):
            fragment = np.array(fragment)
        
        # 确保是2D点云 (N, 2)
        if fragment.shape[1] != 2:
            fragment = fragment[:, :2]  # 只取前两维
        
        # 限制点数
        if len(fragment) > max_points:
            # 等间隔采样
            indices = np.linspace(0, len(fragment)-1, max_points, dtype=int)
            fragment = fragment[indices]
        elif len(fragment) < max_points:
            # 补齐到max_points
            padding_size = max_points - len(fragment)
            if len(fragment) > 0:
                # 重复最后一个点
                last_point = fragment[-1:].repeat(padding_size, axis=0)
                fragment = np.vstack([fragment, last_point])
            else:
                # 如果片段为空，用零点填充
                fragment = np.zeros((max_points, 2))
        
        # 转换为tensor并添加batch维度
        fragment_tensor = torch.FloatTensor(fragment).unsqueeze(0)  # (1, max_points, 2)
        
        return fragment_tensor
    
    def compute_similarity_matrix(self, fragments, batch_size=64, sample_size=None):
        """
        计算所有片段对的相似度矩阵
        
        Args:
            fragments: 片段列表
            batch_size: 批处理大小
            sample_size: 采样片段数量，用于快速测试（None表示使用全部）
            
        Returns:
            numpy.ndarray: 相似度矩阵 (n_fragments, n_fragments)
        """
        print("🔄 计算相似度矩阵...")
        
        # 可选采样以加速测试
        fragment_indices = None
        if sample_size and sample_size < len(fragments):
            print(f"⚡ 快速测试模式：采样 {sample_size} 个片段")
            fragment_indices = np.random.choice(len(fragments), sample_size, replace=False)
            fragments = [fragments[i] for i in fragment_indices]
        
        n_fragments = len(fragments)
        similarity_matrix = np.zeros((n_fragments, n_fragments))
        
        print(f"📊 计算规模: {n_fragments} x {n_fragments} = {n_fragments*n_fragments:,} 个片段对")
        print(f"⏱️  预估时间: ~{(n_fragments * 2.0 / 60):.1f} 分钟")
        
        # 预处理所有片段
        print("   预处理片段...")
        processed_fragments = []
        for i, fragment in enumerate(tqdm(fragments, desc="预处理")):
            processed = self.preprocess_fragment(fragment).to(self.device)
            processed_fragments.append(processed)
        
        # 计算相似度（使用批处理以提高效率）
        print("   计算片段相似度...")
        with torch.no_grad():
            for i in tqdm(range(n_fragments), desc="计算相似度"):
                fragment_i = processed_fragments[i]
                
                # 批处理计算与其他所有片段的相似度
                for start_j in range(0, n_fragments, batch_size):
                    end_j = min(start_j + batch_size, n_fragments)
                    
                    # 创建批次
                    batch_fragments_i = fragment_i.repeat(end_j - start_j, 1, 1)
                    batch_fragments_j = torch.cat([processed_fragments[j] for j in range(start_j, end_j)], dim=0)
                    
                    # 计算匹配概率
                    logits = self.model(batch_fragments_i, batch_fragments_j)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    # 如果只有一个元素，确保是数组
                    if np.isscalar(probs):
                        probs = np.array([probs])
                    
                    # 存储相似度
                    similarity_matrix[i, start_j:end_j] = probs
        
        print("✅ 相似度矩阵计算完成")
        return similarity_matrix, fragment_indices
    
    def evaluate_ndcg(self, similarity_matrix, gt_pairs, fragment_indices=None, save_results=True, output_dir="./results"):
        """
        使用PairingNet的方法评估NDCG
        
        Args:
            similarity_matrix: 相似度矩阵
            gt_pairs: 真实匹配对
            fragment_indices: 采样的片段索引（用于采样模式）
            save_results: 是否保存结果
            output_dir: 结果保存目录
            
        Returns:
            dict: 评估结果
        """
        print("📊 开始NDCG评估...")
        
        # 如果是采样模式，需要过滤GT pairs
        if fragment_indices is not None:
            print(f"🔍 过滤采样片段的GT pairs...")
            # 只保留在采样索引中的GT pairs
            filtered_gt_pairs = []
            index_set = set(fragment_indices)
            for source_idx, target_idx in gt_pairs:
                if source_idx in index_set and target_idx in index_set:
                    # 重新映射索引
                    new_source = list(fragment_indices).index(source_idx)
                    new_target = list(fragment_indices).index(target_idx)
                    filtered_gt_pairs.append([new_source, new_target])
            gt_pairs = filtered_gt_pairs
            print(f"   过滤后GT pairs: {len(gt_pairs)}")
        
        # 使用PairingNet的calculate_NDCG函数
        print("\n=== PairingNet NDCG 评估结果 ===")
        if len(gt_pairs) > 0:
            calculate_NDCG(similarity_matrix, gt_pairs)
        else:
            print("⚠️  采样中没有有效的GT pairs，跳过PairingNet评估")
        
        # 手动计算详细的NDCG结果以获取数值
        print("\n📈 详细NDCG计算...")
        
        # 按照PairingNet的方法创建排序索引
        idx = np.argsort(-similarity_matrix, axis=1)
        length = similarity_matrix.shape[0]
        
        # 创建ground truth和prediction矩阵
        gt = np.zeros((length, length), dtype=np.uint8)
        pred = np.zeros((length, length), dtype=np.uint8)
        
        for i in range(len(gt_pairs)):
            source_idx = gt_pairs[i][0]
            target_idx = gt_pairs[i][1]
            
            # 找到target在source的排序中的位置
            location = np.argwhere(idx[source_idx] == target_idx)[0][0]
            
            # 设置ground truth和prediction
            gt[source_idx, target_idx] = 1
            pred[source_idx, location] = 1
        
        # 移除全零行（没有匹配对的片段）
        def remove_zero_rows(array):
            mask = np.all(array == 0, axis=1)
            return array[~mask]
        
        new_gt = remove_zero_rows(gt)
        new_pred = remove_zero_rows(pred)
        
        # 计算NDCG@5, @10, @20
        ndcg_5 = ndcg(new_gt, new_pred, 5)
        ndcg_10 = ndcg(new_gt, new_pred, 10)
        ndcg_20 = ndcg(new_gt, new_pred, 20)
        
        results = {
            'NDCG@5': float(ndcg_5),
            'NDCG@10': float(ndcg_10),
            'NDCG@20': float(ndcg_20),
            'total_fragments': int(length),
            'total_gt_pairs': int(len(gt_pairs)),
            'fragments_with_matches': int(np.sum(~np.all(gt == 0, axis=1)))
        }
        
        # 显示结果
        print(f"\n📊 EdgeSpark NDCG 评估结果:")
        print(f"   NDCG@5:  {ndcg_5:.4f}")
        print(f"   NDCG@10: {ndcg_10:.4f}")
        print(f"   NDCG@20: {ndcg_20:.4f}")
        print(f"\n📈 数据统计:")
        print(f"   总片段数: {length}")
        print(f"   真实匹配对: {len(gt_pairs)}")
        print(f"   有匹配的片段: {results['fragments_with_matches']}")
        
        # 保存结果
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存JSON结果
            result_file = os.path.join(output_dir, f"ndcg_evaluation_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # 保存相似度矩阵
            matrix_file = os.path.join(output_dir, f"similarity_matrix_{timestamp}.npy")
            np.save(matrix_file, similarity_matrix)
            
            print(f"\n💾 结果已保存:")
            print(f"   - JSON结果: {result_file}")
            print(f"   - 相似度矩阵: {matrix_file}")
        
        return results
    
    def run_evaluation(self, dataset_path, save_results=True, output_dir="./results"):
        """
        运行完整的NDCG评估流程
        
        Args:
            dataset_path: 数据集路径
            save_results: 是否保存结果
            output_dir: 结果保存目录
            
        Returns:
            dict: 评估结果
        """
        print("🚀 开始EdgeSpark NDCG评估")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载数据集
        fragments, gt_pairs = self.load_dataset(dataset_path)
        
        # 3. 计算相似度矩阵
        similarity_matrix, fragment_indices = self.compute_similarity_matrix(fragments, sample_size=100)  # 超快速测试
        
        # 4. 评估NDCG
        results = self.evaluate_ndcg(similarity_matrix, gt_pairs, fragment_indices, save_results, output_dir)
        
        print("\n🎉 评估完成!")
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="EdgeSpark NDCG评估")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--dataset", type=str, required=True, help="数据集文件路径") 
    parser.add_argument("--output", type=str, default="./results", help="结果保存目录")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 创建评估器
    evaluator = EdgeSparkNDCGEvaluator(args.model, device)
    
    # 运行评估
    try:
        results = evaluator.run_evaluation(args.dataset, save_results=True, output_dir=args.output)
        print(f"\n✅ 评估成功完成!")
        print(f"最终NDCG结果: @5={results['NDCG@5']:.4f}, @10={results['NDCG@10']:.4f}, @20={results['NDCG@20']:.4f}")
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        raise

if __name__ == "__main__":
    main()