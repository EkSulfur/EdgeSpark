# EdgeSpark 项目文件结构说明

## 📁 整理后的项目结构

```
EdgeSpark/
├── 📄 PROJECT_README.md              # 项目说明文档
├── 📄 CLAUDE.md                      # AI助手项目指南
├── 📄 EXPERIMENT_SUMMARY.md          # 实验总结和详细分析
├── 📄 FOLDER_STRUCTURE.md            # 本文件 - 文件结构说明
├── 📄 main.py                        # 主程序入口
├── 📄 run_best_model.py              # 快速运行最佳模型
├── 📄 pyproject.toml                 # Python包配置文件
├── 📄 best_final_model.pth           # 最佳训练模型权重
│
├── 📁 dataset/                       # 数据集文件
│   ├── train_set.pkl                 # 训练集
│   ├── valid_set.pkl                 # 验证集
│   └── test_set.pkl                  # 测试集
│
├── 📁 original_code/                 # 原始代码（参考用）
│   ├── network_improved.py          # 原始复杂网络架构
│   ├── train.py                      # 原始训练脚本
│   └── dataset_loader.py            # 原始数据加载器
│
├── 📁 simplified_approach/           # 简化版本实现
│   ├── network_simple.py            # 简化网络架构
│   ├── dataset_simple.py            # 优化的数据处理
│   └── train_simple.py              # 简化训练脚本
│
├── 📁 final_approach/                # 最终最佳实现 ⭐
│   ├── final_approach.py            # 【推荐】最佳网络和训练脚本
│   ├── network_minimal.py           # 极简网络（对比用）
│   └── train_minimal.py             # 极简训练脚本
│
├── 📁 analysis_tools/                # 分析和调试工具
│   ├── debug_data.py                # 数据质量分析
│   ├── quick_test.py                # 快速功能测试
│   ├── run_simple_test.py           # 综合测试套件
│   ├── visualize_results.py         # 结果可视化
│   └── data_analysis.py             # 数据分析工具
│
├── 📁 archive/                       # 归档实验结果
│   └── experiments/                  # 历史训练结果
│       ├── exp_20250716_225043/      # 原始实验
│       ├── simple_exp_20250716_232031/ # 简化版实验
│       └── minimal_exp_20250716_232920/ # 极简版实验
│
├── 📁 Data Generation Code/          # 数据生成代码 (外部项目，不修改)
│   ├── 1_cut_image.py
│   ├── 2_get_gt_pair.py
│   ├── 3_divide_data.py
│   └── ...
│
├── 📁 PairingNet Code/               # 参考实现代码 (外部项目，不修改)
│   ├── run.py
│   ├── matching_test.py
│   └── ...
│
└── 📁 data-analyze/                  # 数据分析脚本 (独立模块)
    ├── check_data.py
    ├── filter_pkl.py
    └── ...
```

## 🎯 各模块功能说明

### 核心实现模块

#### 🏆 final_approach/ - 最佳实现
- **final_approach.py**: 最终最佳模型，准确率60.95%
- **network_minimal.py**: 极简网络架构
- **train_minimal.py**: 极简训练脚本

#### 🔧 simplified_approach/ - 简化实现
- **network_simple.py**: 简化网络架构，准确率59.85%
- **dataset_simple.py**: 优化的数据处理和负采样策略
- **train_simple.py**: 简化训练脚本

#### 📜 original_code/ - 原始代码
- **network_improved.py**: 原始复杂网络（无法收敛）
- **train.py**: 原始训练脚本
- **dataset_loader.py**: 原始数据加载器

### 分析工具模块

#### 🔍 analysis_tools/ - 分析调试工具
- **debug_data.py**: 数据质量分析，发现关键问题
- **quick_test.py**: 快速功能测试
- **run_simple_test.py**: 综合测试套件
- **visualize_results.py**: 结果可视化
- **data_analysis.py**: 数据分析工具

### 支持模块

#### 📊 archive/ - 实验归档
- 存储所有训练实验的结果和日志
- 包含TensorBoard日志和训练历史

#### 🗂️ dataset/ - 数据集
- 训练、验证、测试数据集
- Pickle格式，包含边缘点云和匹配对

### 外部模块（不修改）

#### 📁 Data Generation Code/
- 外部项目的数据生成代码
- 用于生成训练数据集
- **不进行修改**

#### 📁 PairingNet Code/
- 参考实现代码
- 来自相关项目的代码
- **不进行修改**

## 🚀 使用指南

### 快速开始
```bash
# 方法1: 使用快速启动脚本
uv run python run_best_model.py

# 方法2: 直接运行最佳模型
uv run python final_approach/final_approach.py
```

### 分析和调试
```bash
# 数据质量分析
uv run python analysis_tools/debug_data.py

# 快速测试
uv run python analysis_tools/quick_test.py

# 综合测试
uv run python analysis_tools/run_simple_test.py
```

### 不同版本对比
```bash
# 简化版本
uv run python simplified_approach/train_simple.py

# 极简版本
uv run python final_approach/train_minimal.py
```

## 📈 性能对比

| 实现版本 | 文件位置 | 准确率 | 参数量 | 状态 |
|---------|----------|--------|--------|------|
| 原始复杂版 | `original_code/` | 50.0% | ~500K | ❌ 失败 |
| 简化版 | `simplified_approach/` | 59.85% | 224K | ⚠️ 部分成功 |
| 极简版 | `final_approach/network_minimal.py` | 50.0% | 87K | ❌ 失败 |
| **最佳版** | `final_approach/final_approach.py` | **60.95%** | **1.1M** | ✅ **推荐** |

## 🔧 开发工作流

1. **问题分析**: 使用 `analysis_tools/debug_data.py`
2. **快速测试**: 使用 `analysis_tools/quick_test.py`
3. **模型开发**: 在对应的approach文件夹中开发
4. **性能评估**: 使用 `analysis_tools/visualize_results.py`
5. **结果归档**: 保存到 `archive/experiments/`

## 📝 文档说明

- **PROJECT_README.md**: 项目整体说明
- **EXPERIMENT_SUMMARY.md**: 详细的实验分析和结论
- **CLAUDE.md**: AI助手使用指南
- **FOLDER_STRUCTURE.md**: 本文件，文件结构说明

## 🎯 推荐使用路径

1. **新用户**: 阅读 `PROJECT_README.md` → 运行 `run_best_model.py`
2. **开发者**: 阅读 `EXPERIMENT_SUMMARY.md` → 使用 `final_approach/final_approach.py`
3. **研究者**: 比较各个approach文件夹中的不同实现
4. **调试者**: 使用 `analysis_tools/` 中的工具进行分析

---

*项目整理完成时间: 2025年7月16日*