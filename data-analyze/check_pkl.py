import os
import pickle
import sys

def print_pkl_structure(filepath):
    """
    打印 PKL 文件的结构
    
    参数:
        filepath: PKL 文件路径
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"文件: {filepath}")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print("字典键:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")
                
                # 通用方式检查是否具有shape属性
                shape_attr = getattr(data[key], 'shape', None)
                if shape_attr:
                    print(f"    形状: {shape_attr}")
                elif isinstance(data[key], list):
                    print(f"    长度: {len(data[key])}")
                    if len(data[key]) > 0:
                        print(f"    第一个元素类型: {type(data[key][0])}")
                        
                        # 通用方式检查是否具有shape属性
                        first_elem_shape = getattr(data[key][0], 'shape', None)
                        if first_elem_shape:
                            print(f"    第一个元素形状: {first_elem_shape}")
        else:
            print("不是字典类型")
            
            # 通用方式检查是否具有shape属性
            shape_attr = getattr(data, 'shape', None)
            if shape_attr:
                print(f"形状: {shape_attr}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果提供了命令行参数，使用它作为文件路径
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
        print_pkl_structure(pkl_path)
    else:
        # 否则使用硬编码的路径
        pkl_path = "/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset_backup/valid_set_with_downsample.pkl"
        print_pkl_structure(pkl_path)