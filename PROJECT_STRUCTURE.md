# EdgeSpark Project Structure

## 📁 项目文件组织结构

```
EdgeSpark/
├── 📄 README.md                           # 项目主要说明
├── 📄 CLAUDE.md                           # Claude Code 项目指令
├── 📄 PROJECT_STRUCTURE.md                # 本文件 - 项目结构说明
│
├── 📁 dataset/                            # 数据集目录
│   ├── train_set.pkl                      # 训练数据集
│   ├── valid_set.pkl                      # 验证数据集
│   └── test_set.pkl                       # 测试数据集
│
├── 📁 config/                             # 配置文件
│   ├── pyproject.toml                     # uv 包管理配置
│   ├── requirments.yaml                   # conda 环境配置
│   └── uv.lock                           # uv 锁定文件
│
├── 📁 src/                                # 核心源代码
│   ├── 📁 core/                           # 核心模块
│   │   └── main.py                        # 主入口文件
│   │
│   ├── 📁 data/                           # 数据处理模块
│   │   ├── dataset_loader.py              # 基础数据加载器
│   │   ├── dataset_simple.py              # 简化数据加载器
│   │   ├── improved_dataset_loader.py     # 改进数据加载器
│   │   └── pairing_inspired_dataloader.py # PairingNet风格数据加载器 ⭐
│   │
│   ├── 📁 models/                         # 模型定义
│   │   ├── network_simple.py              # 简单网络模型
│   │   ├── network_minimal.py             # 最小网络模型
│   │   ├── network_improved.py            # 改进网络模型
│   │   ├── final_approach.py              # 最终方法
│   │   ├── improved_final_approach.py     # 改进最终方法(FocalLoss+温度缩放)
│   │   ├── pairing_inspired_model.py      # PairingNet风格模型 ⭐
│   │   ├── enhanced_feature_extractor.py  # 增强特征提取器
│   │   ├── hybrid_network.py              # 混合网络
│   │   └── quick_enhanced_test.py         # 快速增强测试
│   │
│   ├── 📁 training/                       # 训练模块
│   │   ├── train.py                       # 基础训练脚本
│   │   ├── train_simple.py                # 简单训练脚本
│   │   ├── train_minimal.py               # 最小训练脚本
│   │   └── hybrid_train.py                # 混合训练脚本
│   │
│   ├── 📁 experiments/                    # 实验脚本
│   │   ├── pairing_inspired_experiment.py # PairingNet风格实验 ⭐
│   │   └── quick_pairing_test.py          # 快速PairingNet测试 ⭐
│   │
│   └── 📁 analysis/                       # 分析工具
│       ├── comprehensive_diagnosis.py     # 综合诊断
│       └── feature_analysis_summary.py    # 特征分析摘要
│
├── 📁 tools/                              # 辅助工具
│   ├── 📁 analysis/                       # 分析工具
│   │   ├── check_data.py                  # 数据检查
│   │   ├── check_pkl.py                   # PKL文件检查
│   │   ├── data_analysis.py               # 数据分析
│   │   └── visualize_results.py           # 结果可视化
│   │
│   └── 📁 scripts/                        # 脚本工具
│       ├── run_quick_test.sh              # 快速测试脚本
│       └── run_training.sh                # 训练脚本
│
├── 📁 results/                            # 结果存储 🆕
│   ├── 📁 models/                         # 训练好的模型
│   │   ├── best_final_model_ordered.pth   # 有序采样最佳模型
│   │   ├── best_final_model_random.pth    # 随机采样最佳模型
│   │   ├── best_final_model_padding.pth   # 填充采样最佳模型
│   │   ├── best_enhanced_model_*.pth      # 增强模型系列
│   │   └── best_improved_model.pth        # 改进模型
│   │
│   ├── 📁 experiments/                    # 实验结果
│   │   ├── improved_dataloader_results_*.json     # 改进数据加载器结果
│   │   ├── enhanced_feature_results_*.json        # 增强特征结果
│   │   └── diagnosis_report_*.json                # 诊断报告
│   │
│   └── 📁 analysis/                       # 分析结果
│       ├── comprehensive_analysis_report.md       # 综合分析报告 ⭐
│       ├── enhanced_features.png          # 增强特征可视化
│       ├── simple_features.png            # 简单特征可视化
│       └── feature_visualizations/        # 特征可视化目录
│
├── 📁 experiments/                        # 历史实验(保留)
│   ├── 📁 evaluation/                     # 评估脚本
│   ├── 📁 runners/                        # 运行脚本
│   └── 📁 testing/                        # 测试脚本
│
├── 📁 archive/                            # 历史存档
│   └── 📁 experiments/                    # 历史实验结果
│       ├── exp_20250716_225043/           # 实验存档
│       ├── simple_exp_*/                  # 简单实验系列
│       └── minimal_exp_*/                 # 最小实验系列
│
├── 📁 docs/                               # 文档目录
│   ├── PROJECT_README.md                  # 项目说明
│   ├── EXPERIMENT_SUMMARY.md              # 实验总结
│   ├── FINAL_IMPROVEMENT_RESULTS.md       # 最终改进结果
│   └── *.md                              # 其他文档
│
├── 📁 assets/                             # 资源文件
│   ├── 📁 Pictures/                       # 图片资源
│   └── fragment_visualization.png         # 碎片可视化
│
├── 📁 Data Generation Code/               # 数据生成代码(外部)
├── 📁 PairingNet Code/                    # PairingNet参考代码(外部) ⭐
└── 📁 feature_visualizations/             # 特征可视化结果
```

## 🔑 关键文件说明

### ⭐ 核心改进文件（基于PairingNet）

1. **`src/data/pairing_inspired_dataloader.py`**
   - 基于PairingNet的数据加载器
   - 实现邻接矩阵构建、空间特征提取
   - 支持PairingNet风格的点云归一化

2. **`src/models/pairing_inspired_model.py`**
   - 基于PairingNet的模型架构
   - 图卷积网络 + 空间注意力机制
   - FocalLoss + 温度缩放相似度计算

3. **`src/experiments/pairing_inspired_experiment.py`**
   - 综合PairingNet风格实验脚本
   - 多配置对比：邻接矩阵k值、空间特征、温度参数等

4. **`results/analysis/comprehensive_analysis_report.md`**
   - 完整的实验结果分析报告
   - 性能瓶颈诊断和改进建议

### 📊 实验结果文件

1. **`results/experiments/improved_dataloader_results_*.json`**
   - 改进数据加载器的实验结果
   - ordered > random > padding 的效果排序

2. **`results/experiments/diagnosis_report_*.json`**
   - 详细的项目诊断报告
   - 数据质量、模型行为、任务复杂性分析

### 🎯 模型文件组织

- **基础模型系列**: `network_simple.py`, `network_minimal.py`
- **改进模型系列**: `final_approach.py`, `improved_final_approach.py`
- **PairingNet风格**: `pairing_inspired_model.py` ⭐
- **特殊方法**: `fourier_approach.py`, `hybrid_network.py`

## 📈 开发时间线

### Phase 1: 基础实现 (2025-07-16)
- 实现基础网络和数据加载器
- 完成简单实验验证

### Phase 2: 性能诊断 (2025-07-17 早期)
- 发现模型性能瓶颈问题
- 进行全面的数据和模型分析

### Phase 3: PairingNet启发改进 (2025-07-17 后期) ⭐
- 分析PairingNet代码架构
- 实现基于邻接矩阵的空间建模
- 添加FocalLoss和温度缩放等技术

### Phase 4: 实验验证 (进行中)
- 运行PairingNet风格的综合实验
- 验证各项改进的效果

## 🚀 下一步计划

1. **完成PairingNet风格实验**: 验证邻接矩阵+空间特征的效果
2. **修复技术问题**: 解决PyTorch兼容性问题
3. **深度优化**: 引入更多PairingNet的成功经验
4. **对比学习**: 实现InfoNCE等对比学习损失函数

## 📝 使用说明

### 快速开始
```bash
# 安装依赖
uv sync

# 运行PairingNet风格快速测试
uv run python src/experiments/quick_pairing_test.py

# 运行完整实验
uv run python src/experiments/pairing_inspired_experiment.py
```

### 主要命令
```bash
# 数据分析
uv run python tools/analysis/data_analysis.py

# 模型训练
uv run python src/training/train.py

# 结果可视化
uv run python tools/analysis/visualize_results.py
```

---

**⚡ 项目亮点**: EdgeSpark通过深入分析PairingNet的成功经验，实现了基于邻接矩阵的空间关系建模，这是2D碎片匹配领域的重要改进方向。