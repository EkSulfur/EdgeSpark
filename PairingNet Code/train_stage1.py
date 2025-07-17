#!/usr/bin/env python3
"""
训练STAGE_ONE模型的脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import __init__
import torch
import math
from utils import pipeline, config
from PairingNet_train_val_test import Train_model

def main():
    # 获取配置参数
    opt = config.args
    
    # 设置模型类型为匹配训练
    opt.model_type = 'matching_train'
    
    # 设置温度系数
    temp = math.sqrt(opt.feature_dim)
    
    # 设置网络
    net = pipeline.Vanilla
    
    # 设置实验名称
    exp_name = 'stage1_contour_extraction'
    
    print("开始训练STAGE_ONE模型...")
    print(f"数据集路径:")
    print(f"  训练集: {opt.train_set}")
    print(f"  验证集: {opt.valid_set}")
    print(f"  测试集: {opt.test_set}")
    print(f"实验名称: {exp_name}")
    print(f"温度系数: {temp}")
    
    # 创建训练器
    trainer = Train_model(net, opt, temp, exp_name)
    
    # 开始训练
    trainer.train_start()
    
    print("STAGE_ONE模型训练完成！")
    print(f"模型保存路径: ./EXP/{exp_name}/checkpoint/")

if __name__ == "__main__":
    main()