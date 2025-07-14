import pickle
import os

def filter_and_save_pkl(input_path, output_path, keys_to_keep):
    """
    加载一个 PKL 文件，筛选指定的键，然后将结果保存到新的 PKL 文件。

    参数:
        input_path (str): 原始 PKL 文件的路径。
        output_path (str): 保存筛选后数据的新 PKL 文件路径。
        keys_to_keep (list): 需要保留的字典键列表。
    """
    print(f"正在从 '{input_path}' 加载数据...")
    
    try:
        # 使用 'rb' (read binary) 模式打开文件
        with open(input_path, 'rb') as f:
            original_data = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{input_path}'")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    # 检查加载的数据是否为字典
    if not isinstance(original_data, dict):
        print("错误: 加载的数据不是字典类型，无法按键筛选。")
        return
        
    print("数据加载成功。开始筛选键...")
    
    # 创建一个新的空字典来存放筛选后的数据
    filtered_data = {}
    
    # 遍历需要保留的键
    for key in keys_to_keep:
        # 检查原始数据中是否存在这个键
        if key in original_data:
            print(f"  - 保留键: '{key}'")
            # 如果存在，将其键值对复制到新字典中
            filtered_data[key] = original_data[key]
        else:
            # 如果原始数据中不存在这个键，打印一个警告信息
            print(f"  - 警告: 键 '{key}' 在原始文件中未找到，将被忽略。")
            
    print("\n筛选完成。")
    print(f"正在将筛选后的数据保存到 '{output_path}'...")
    
    try:
        # 使用 'wb' (write binary) 模式打开新文件
        with open(output_path, 'wb') as f:
            # 将新字典序列化到新文件中
            # protocol=pickle.HIGHEST_PROTOCOL 可以提高效率和压缩率
            pickle.dump(filtered_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("文件保存成功！")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == "__main__":
    # 1. 设置原始文件路径
    original_pkl_path = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset_backup/train_set_with_downsample.pkl"
    original_pkl_path_valid = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset_backup/valid_set_with_downsample.pkl"
    original_pkl_path_test = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset_backup/test_set_with_downsample.pkl"
    
    
    # 2. 自动生成输出文件路径
    output_pkl_path = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset/train_set_with_downsample.pkl"
    output_pkl_path_valid = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset/valid_set_with_downsample.pkl"
    output_pkl_path_test = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset/test_set_with_downsample.pkl"
    
    # 3. 定义您想保留的字典键列表
    keys_to_preserve = [
        'full_pcd_all', 
        'shape_all', 
        'GT_pairs', 
        'source_ind', 
        'target_ind'
    ]
    
    # 4. 调用函数执行筛选和保存
    # filter_and_save_pkl(original_pkl_path, output_pkl_path, keys_to_preserve)
    filter_and_save_pkl(original_pkl_path_valid, output_pkl_path_valid, keys_to_preserve)
    filter_and_save_pkl(original_pkl_path_test, output_pkl_path_test, keys_to_preserve)
