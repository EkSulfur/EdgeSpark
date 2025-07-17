# EdgeSpark 项目文件结构

## 📁 目录结构

```
EdgeSpark/
├── src/                      # 源代码目录
│   ├── core/                 # 核心模块
│   │   └── main.py          # 主入口文件
│   ├── models/              # 模型定义
│   │   ├── network_improved.py     # 原始复杂网络
│   │   ├── network_simple.py       # 简化网络
│   │   ├── final_approach.py       # 最佳方案网络
│   │   ├── hybrid_network.py       # 混合网络
│   │   ├── improved_approach.py    # 改进特征工程
│   │   ├── high_sampling_approach.py # 高采样方案
│   │   └── fourier_approach.py     # 傅里叶变换方案
│   ├── data/                # 数据处理模块
│   │   ├── dataset_loader.py       # 原始数据加载器
│   │   └── dataset_simple.py       # 简化数据加载器
│   └── training/            # 训练脚本
│       ├── train.py                # 原始训练脚本
│       ├── train_simple.py         # 简化训练脚本
│       ├── train_minimal.py        # 最小训练脚本
│       └── hybrid_train.py         # 混合方法训练
├── experiments/             # 实验相关
│   ├── evaluation/          # 评估脚本
│   │   ├── final_evaluation_fixed.py
│   │   ├── improved_final_evaluation.py
│   │   ├── quick_final_evaluation.py
│   │   └── comprehensive_evaluation.py
│   ├── testing/             # 测试脚本
│   │   ├── test_final_approaches.py
│   │   ├── test_hybrid_quick.py
│   │   ├── test_improved_approach.py
│   │   └── quick_improved_test.py
│   └── runners/             # 运行脚本
│       ├── run_best_model.py
│       ├── run_hybrid_experiment.py
│       └── hybrid_full_test.py
├── tools/                   # 工具和脚本
│   ├── analysis/            # 分析工具
│   │   ├── data_analysis.py
│   │   ├── debug_data.py
│   │   ├── quick_test.py
│   │   ├── visualize_results.py
│   │   ├── check_data.py
│   │   ├── check_image.py
│   │   ├── check_pkl.py
│   │   └── filter_pkl.py
│   └── scripts/             # Shell脚本
│       ├── run_quick_test.sh
│       └── run_training.sh
├── docs/                    # 文档
│   ├── README.md                    # 主文档
│   ├── CLAUDE.md                    # Claude指令
│   ├── PROJECT_README.md            # 项目说明
│   ├── EXPERIMENT_SUMMARY.md        # 实验总结
│   ├── FINAL_IMPROVEMENT_RESULTS.md # 最终改进结果
│   ├── HYBRID_APPROACH_PLAN.md      # 混合方法计划
│   ├── HYBRID_EXPERIMENT_RESULTS.md # 混合实验结果
│   └── FOLDER_STRUCTURE.md          # 文件结构说明
├── config/                  # 配置文件
│   ├── pyproject.toml              # Python项目配置
│   ├── requirments.yaml            # Conda环境
│   └── uv.lock                     # UV锁定文件
├── assets/                  # 静态资源
│   ├── Pictures/                   # 图片
│   ├── fragment_visualization.png  # 可视化图
│   └── best_final_model.pth       # 最佳模型
├── dataset/                 # 数据集 (保持不变)
│   ├── train_set.pkl
│   ├── valid_set.pkl
│   └── test_set.pkl
├── archive/                 # 归档 (保持不变)
│   └── experiments/               # 历史实验记录
├── Data Generation Code/     # 外部数据生成代码 (保持不变)
└── PairingNet Code/         # 外部参考代码 (保持不变)
```

## 🎯 目录说明

### src/ - 源代码
**核心开发代码，按功能模块组织**
- `core/`: 主入口和核心逻辑
- `models/`: 所有神经网络模型定义
- `data/`: 数据加载和预处理
- `training/`: 训练相关脚本

### experiments/ - 实验代码
**实验、评估、测试相关代码**
- `evaluation/`: 性能评估脚本
- `testing/`: 功能测试脚本  
- `runners/`: 实验运行脚本

### tools/ - 工具集
**辅助工具和分析脚本**
- `analysis/`: 数据分析和可视化
- `scripts/`: Shell脚本和自动化工具

### docs/ - 文档
**所有项目文档集中管理**
- 实验报告、技术文档、使用说明

### config/ - 配置
**项目配置文件**
- 依赖管理、环境配置

### assets/ - 静态资源
**模型文件、图片等静态资源**

### 其他目录 (保持现有结构)
- `dataset/`: 数据集文件
- `archive/`: 历史实验记录
- `Data Generation Code/`: 外部数据生成代码
- `PairingNet Code/`: 外部参考代码

## 🔄 迁移建议

### 推荐迁移步骤:
1. **备份当前项目**: `cp -r EdgeSpark EdgeSpark_backup`
2. **创建新结构**: 按上述目录创建文件夹
3. **逐步迁移**: 使用 `git mv` 移动文件保持版本历史
4. **更新引用**: 修改import路径和相对路径引用
5. **测试验证**: 确保所有功能正常工作

### 示例迁移命令:
```bash
# 移动模型文件
git mv original_code/network_improved.py src/models/
git mv simplified_approach/network_simple.py src/models/
git mv final_approach/final_approach.py src/models/

# 移动训练脚本
git mv original_code/train.py src/training/
git mv simplified_approach/train_simple.py src/training/

# 移动文档
git mv *.md docs/
```

## 💡 使用指南

### 开发时:
- 新模型 → `src/models/`
- 新训练脚本 → `src/training/` 
- 数据处理 → `src/data/`

### 实验时:
- 评估脚本 → `experiments/evaluation/`
- 测试代码 → `experiments/testing/`
- 运行脚本 → `experiments/runners/`

### 分析时:
- 分析工具 → `tools/analysis/`
- 自动化脚本 → `tools/scripts/`

---

*此结构遵循Python项目最佳实践，便于开发、维护和协作*
