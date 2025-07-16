# 记录

## 关于数据集
数据集路径/home/liuchenghao/Lab_disk/liuchenghao/EdgeSpark/dataset/
分为train, valid, test三个部分
保存为pkl文件

### 数据集包含的项
- 'full_pcd_all'：每个碎片的边缘点云
- 'GT_pairs'：匹配的碎片对
- 'source_ind'：匹配的碎片对中源碎片匹配的点的索引
- 'target_ind'：匹配的碎片对中目标碎片匹配的点的索引 

其中后两个可能在实际训练中不会用到