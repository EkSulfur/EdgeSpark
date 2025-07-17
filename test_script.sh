#!/bin/bash

# 简单测试脚本功能
echo "测试脚本基本功能..."

# 测试帮助信息
echo "1. 测试帮助信息："
./run_contour_extraction.sh -h

echo -e "\n2. 测试状态查看："
./run_contour_extraction.sh -s

echo -e "\n3. 检查脚本是否可执行："
ls -la run_contour_extraction.sh

echo -e "\n4. 检查项目目录结构："
ls -la "PairingNet Code/"

echo -e "\n测试完成！"