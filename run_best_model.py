#!/usr/bin/env python3
"""
EdgeSpark - 运行最佳模型
快速启动脚本，使用最佳配置训练模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'final_approach'))

from final_approach.final_approach import main

if __name__ == "__main__":
    print("🚀 EdgeSpark - 运行最佳模型")
    print("=" * 50)
    print("使用最佳配置训练边缘形状匹配网络")
    print("预期准确率: ~61%")
    print("=" * 50)
    
    main()