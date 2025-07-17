import pickle
import numpy as np
from PIL import Image

with open('./Fragments dataset/real_dataset_all.pkl', 'rb') as f:
    data = pickle.load(f)

print("数据类型:", type(data))
print("\n各键的第一个数据:")
for key in data.keys():
    value = data[key]
    print(f"\n{key}:")
    if isinstance(value, (list, np.ndarray)):
        print(f"类型: {type(value)}")
        print(f"长度/形状: {len(value) if isinstance(value, list) else value.shape}")
        if len(value) > 0:
            first_elem = value[0]
            if isinstance(first_elem, np.ndarray):
                print(f"第一个元素形状: {first_elem.shape}")
                print(f"第一个元素数据类型: {first_elem.dtype}")
            else:
                print(f"第一个元素: {first_elem}")
    else:
        print(f"类型: {type(value)}")
        print("值:", value)