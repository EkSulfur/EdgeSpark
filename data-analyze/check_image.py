import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. 加载 pkl 文件
with open("C:\\code\\C_repos\\EdgeSpark\\dataset\\small\\ori_dataset_all.pkl", 'rb') as f:
    data = pickle.load(f)

# 2. 检查图像数据结构
img_data = data['img_all']
print(f"图像数据类型: {type(img_data)}")
print(f"图像数量: {len(img_data)}")
if len(img_data) > 0:
    print(f"第一个图像形状: {img_data[0].shape}")

# 3. 创建保存图像的文件夹
os.makedirs("extracted_images", exist_ok=True)

# 4. 提取并保存图像
for i, img in enumerate(img_data):
    if i >= 10:  # 只保存前10张图像
        break
    
    # 如果图像是 RGB 格式 (H, W, 3)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 确保数值范围在 0-255
        if img.dtype == np.float32 or img.dtype == np.float64:
            if np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # 保存图像
        Image.fromarray(img).save(f"extracted_images/image_{i}.png")
    
    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"图像 {i}")
    plt.axis('off')
    plt.show()

print("图像提取完成，已保存到 'extracted_images' 文件夹")