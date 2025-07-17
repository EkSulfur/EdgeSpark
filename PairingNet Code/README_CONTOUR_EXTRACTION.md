# PairingNet Contour特征提取模块使用指南

## 概述
本指南帮助您复现PairingNet模型并提取其中的contour特征提取模块，使其可以独立应用于其他项目。

## 文件结构
```
PairingNet Code/
├── dataset/                              # 数据集文件夹
│   ├── train_set_with_downsample.pkl
│   ├── valid_set_with_downsample.pkl
│   └── test_set_with_downsample.pkl
├── train_stage1.py                       # 训练STAGE_ONE模型
├── contour_feature_extractor.py          # 独立的contour特征提取模块
├── extract_contour_module.py             # 提取模块的完整脚本
└── README_CONTOUR_EXTRACTION.md          # 本说明文档
```

## 使用步骤

### 1. 准备数据集
确保在 `dataset/` 文件夹下有以下三个文件：
- `train_set_with_downsample.pkl`
- `valid_set_with_downsample.pkl`
- `test_set_with_downsample.pkl`

### 2. 训练STAGE_ONE模型
```bash
cd "PairingNet Code"
python train_stage1.py
```

训练完成后，模型检查点将保存在 `./EXP/stage1_contour_extraction/checkpoint/` 目录中。

### 3. 提取contour特征模块
```bash
python extract_contour_module.py
```

这将生成独立的contour特征提取模块文件：`contour_feature_extractor_stage1_contour_extraction.pth`

### 4. 在其他项目中使用

#### 4.1 加载模块
```python
from contour_feature_extractor import load_contour_extractor
import torch

# 加载模型
extractor, args = load_contour_extractor('contour_feature_extractor_stage1_contour_extraction.pth')
extractor.eval()
```

#### 4.2 准备输入数据
```python
inputs = {
    'pcd': torch.randn(1, 100, 2),      # contour点云数据 [batch_size, num_points, 2]
    'c_input': torch.randn(1, 100, 3, 7), # contour输入 [batch_size, num_points, channels, patch_size]
    'adj': torch.randn(2, 1000)         # 邻接矩阵 [2, num_edges]
}
```

#### 4.3 提取特征
```python
with torch.no_grad():
    l_c = extractor(inputs)  # 输出contour特征
    print(f'Contour特征shape: {l_c.shape}')  # [batch_size, num_points, feature_dim]
```

## 核心模块说明

### ContourFeatureExtractor类
这是从PairingNet的Vanilla类中提取的独立contour特征提取模块，包含：

- **flatten_net**: 将patch特征展平的网络
- **encoder_c**: 基于GCN的contour编码器
- **fc**: 全连接层，用于特征维度调整

### 输入格式
- `pcd`: contour点云数据，形状为 `[batch_size, num_points, 2]`
- `c_input`: contour输入数据，形状为 `[batch_size, num_points, channels, patch_size]`
- `adj`: 图的邻接矩阵，形状为 `[2, num_edges]`

### 输出格式
- `l_c`: contour特征，形状为 `[batch_size, num_points, feature_dim]`

## 注意事项

1. **环境依赖**: 确保已安装PyTorch和相关依赖包
2. **GPU支持**: 代码中使用了CUDA操作，需要GPU环境
3. **数据格式**: 输入数据需要符合PairingNet的格式要求
4. **模型参数**: 提取的模块保留了原始训练的参数配置

## 故障排除

### 常见问题
1. **找不到检查点文件**: 确保已完成STAGE_ONE模型的训练
2. **CUDA错误**: 检查GPU环境配置
3. **数据格式错误**: 确保输入数据格式正确

### 日志位置
- 训练日志: `./EXP/stage1_contour_extraction/summary/`
- 模型检查点: `./EXP/stage1_contour_extraction/checkpoint/`

## 技术细节

### 特征提取流程
1. 对contour点云进行预处理（归一化）
2. 通过flatten_net处理patch特征
3. 结合contour和patch特征
4. 通过GCN编码器提取最终特征

### 与原始代码的对应关系
- 对应pipeline.py中Vanilla类的forward函数第103-125行
- 提取了与contour相关的网络组件
- 保持了原始的前向传播逻辑

---

如有问题，请检查环境配置或查看错误日志。