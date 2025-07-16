#!/bin/bash

# EdgeSpark 训练脚本
# 使用方法: bash run_training.sh

echo "EdgeSpark 2D碎片匹配训练脚本"
echo "================================"

# 1. 更新环境
echo "1. 更新Python环境..."
uv sync

if [ $? -ne 0 ]; then
    echo "环境更新失败，请检查依赖配置"
    exit 1
fi

echo "环境更新完成"

# 2. 运行训练
echo ""
echo "2. 开始训练模型..."
uv run python train.py

if [ $? -ne 0 ]; then
    echo "训练失败，请检查错误信息"
    exit 1
fi

echo ""
echo "3. 训练完成！"

# 3. 查找最新的实验目录
LATEST_EXP=$(ls -td experiments/exp_* 2>/dev/null | head -n1)

if [ -n "$LATEST_EXP" ]; then
    echo "最新实验目录: $LATEST_EXP"
    echo ""
    echo "4. 生成实验报告..."
    uv run python visualize_results.py "$LATEST_EXP"
    
    echo ""
    echo "============================================"
    echo "训练完成！实验结果保存在: $LATEST_EXP"
    echo "============================================"
    echo "主要文件："
    echo "  - best_model.pth: 最佳模型权重"
    echo "  - training_history.json: 训练历史数据"
    echo "  - training_history.png: 训练过程可视化"
    echo "  - confusion_matrix.png: 混淆矩阵"
    echo "  - probability_analysis.png: 概率分析"
    echo "  - evaluation_report.txt: 详细评估报告"
    echo ""
    echo "要查看TensorBoard日志，运行："
    echo "  tensorboard --logdir=$LATEST_EXP/tensorboard"
else
    echo "未找到实验目录，可能训练过程中出现了问题"
fi