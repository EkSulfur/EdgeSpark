# EdgeSpark - 2D Fragment Matching Project

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

EdgeSpark是一个基于深度学习的2D碎片匹配项目，旨在通过深度学习技术匹配不规则2D碎片，寻找可能相邻的碎片对。

## 🎯 项目概述

### 核心任务
使用暴力采样方法结合深度学习进行2D碎片匹配，判断两个碎片是否可能相邻拼接。

### 技术特点
- **边缘点云处理**: 处理不规则2D碎片的边缘点云数据
- **随机采样策略**: 通过足够多的随机采样减少对齐问题的影响  
- **Transformer架构**: 使用Transformer处理n1×n2序列进行匹配
- **端到端训练**: 使用二元交叉熵损失进行端到端碎片匹配

## 🏗️ 项目结构

```
EdgeSpark/
├── src/                     # 核心源代码
│   ├── core/               # 主入口模块
│   ├── models/             # 神经网络模型
│   ├── data/               # 数据处理模块
│   └── training/           # 训练脚本
├── experiments/            # 实验相关代码
│   ├── evaluation/         # 性能评估
│   ├── testing/           # 功能测试
│   └── runners/           # 实验运行脚本
├── tools/                  # 工具和分析脚本
│   ├── analysis/          # 数据分析工具
│   └── scripts/           # 自动化脚本
├── docs/                   # 项目文档
├── config/                 # 配置文件
├── assets/                 # 静态资源
├── dataset/               # 数据集文件
└── archive/               # 历史实验记录
```

详细结构说明请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🚀 快速开始

### 环境要求
- Python ≥ 3.13
- PyTorch ≥ 2.0
- CUDA 11.8+ (推荐)
- UV包管理器 (推荐)

### 安装依赖

#### 使用UV (推荐)
```bash
# 克隆项目
git clone <repository-url>
cd EdgeSpark

# 安装依赖
uv sync
```

#### 使用Conda (传统方式)
```bash
# 创建conda环境
conda env create -f config/requirments.yaml
conda activate edgespark
```

### 数据准备
确保数据集文件位于 `dataset/` 目录:
```
dataset/
├── train_set.pkl    # 训练数据 (7,308样本)
├── valid_set.pkl    # 验证数据 (274样本)  
└── test_set.pkl     # 测试数据 (4,740样本)
```

### 运行示例

#### 训练最佳模型
```bash
uv run python src/training/train_minimal.py
```

#### 快速评估
```bash
uv run python experiments/evaluation/quick_final_evaluation.py
```

#### 运行最佳模型
```bash
uv run python experiments/runners/run_best_model.py
```

## 📊 模型性能

### 历史最佳结果

| 模型方案 | 准确率 | F1-Score | AUC | 参数量 | 状态 |
|----------|--------|----------|-----|--------|------|
| **final_approach** | **60.95%** | 60.00% | 65.00% | 1.1M | 🏆 当前最佳 |
| 简化网络 | 59.85% | 58.00% | 62.00% | 224K | ✅ 稳定 |
| 高采样方案 | 57.30% | 51.05% | 57.02% | 413K | 🔬 实验性 |
| 傅里叶方案 | 56.20% | 50.41% | 59.03% | 386K | 🔬 实验性 |
| 原始复杂网络 | 50.00% | - | - | ~500K | ❌ 无法收敛 |

### 性能分析
- **最佳方案**: final_approach在60.95%准确率下表现最佳
- **效率平衡**: 在性能和计算效率间取得良好平衡
- **稳定性**: 训练过程稳定，泛化能力强

## 🧠 核心算法

### 暴力采样方法
1. **随机采样**: 从两个碎片中分别采样n1、n2个边缘段
2. **特征编码**: 对采样段进行特征编码
3. **相似度计算**: 计算n1×n2×d的相似度矩阵
4. **深度匹配**: 使用深度网络判断碎片对匹配概率

### 网络架构
- **EdgeShapeEncoder**: 专门的边缘形状编码器
- **特征融合**: 多种相似度度量的组合
- **二分类输出**: 输出匹配概率

## 🔬 实验记录

### 已完成实验
1. **原始网络调试** - 解决收敛问题
2. **网络简化优化** - 提升到59.85%准确率
3. **最佳方案开发** - 达到60.95%准确率
4. **混合方法探索** - 多采样策略实验
5. **高级特征工程** - 高采样和傅里叶变换方法

### 关键发现
- **数据质量是瓶颈**: 特征空间可分性有限
- **简单有效原则**: 过度复杂化反而降低性能
- **采样策略局限**: 多采样未能显著提升性能

详细实验报告:
- [实验总结](docs/EXPERIMENT_SUMMARY.md)
- [最终改进结果](docs/FINAL_IMPROVEMENT_RESULTS.md)
- [混合方法实验](docs/HYBRID_EXPERIMENT_RESULTS.md)

## 🛠️ 开发指南

### 添加新模型
1. 在 `src/models/` 中创建新的模型文件
2. 继承适当的基类
3. 在 `src/training/` 中创建对应的训练脚本
4. 在 `experiments/` 中添加评估代码

### 运行实验
```bash
# 功能测试
uv run python experiments/testing/test_final_approaches.py

# 性能评估  
uv run python experiments/evaluation/final_evaluation_fixed.py

# 数据分析
uv run python tools/analysis/data_analysis.py
```

### 代码规范
- 遵循PEP 8代码风格
- 添加适当的类型注解
- 编写清晰的文档字符串
- 保持模块化和可测试性

## 📈 性能优化建议

### 短期优化 (2-5%提升)
- **数据清洗**: 去除噪声和错误标注
- **特征工程**: 设计更好的几何特征
- **训练策略**: 使用Focal Loss等高级损失函数

### 长期突破 (10-20%提升)  
- **图神经网络**: 将碎片建模为图结构
- **自监督学习**: 无监督预训练提取更好特征
- **多模态融合**: 结合其他类型的碎片信息

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- 感谢Claude Code在项目开发中的技术支持
- 参考了PairingNet等相关研究工作
- 使用了PyTorch深度学习框架

## 📞 联系方式

如有问题或建议，请创建Issue或联系项目维护者。

---

*EdgeSpark - 让碎片匹配变得智能* 🧩✨