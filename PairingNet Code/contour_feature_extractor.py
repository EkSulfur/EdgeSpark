#!/usr/bin/env python3
"""
独立的contour特征提取模块
"""
import torch
import torch.nn as nn
from utils.encoder import FlattenNet_average, MyDeepGCN

class ContourFeatureExtractor(nn.Module):
    """
    从PairingNet中提取的contour特征提取模块
    用于输入contour点云数据，输出对应的l_c特征
    """
    
    def __init__(self, args):
        super(ContourFeatureExtractor, self).__init__()
        
        # 从Vanilla类中提取的组件
        self.flatten_net = FlattenNet_average(args.flattenNet_config)
        self.encoder_c = MyDeepGCN(args)
        self.fc = nn.Linear(args.in_channels+2, args.in_channels)
        self.c_feature = args.flattenNet_config["output_dim"]
        
    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: dict包含以下键值:
                - 'pcd': contour点云数据 [bs, n, 2]
                - 'c_input': contour输入 [bs, n, ?, ?]
                - 'adj': 邻接矩阵
        Returns:
            l_c: contour特征 [bs, n, feature_dim]
        """
        contour = inputs['pcd']
        c_input = inputs['c_input']
        adj = inputs['adj']
        
        bs, n, _, _ = c_input.shape
        c_feature = self.c_feature
        
        # 对contour进行预处理（与原始代码保持一致）
        contour = contour + torch.tensor([1, 1]).cuda()
        
        # 展平c_input
        flatted_c = self.flatten_net(c_input)
        flatted_c = flatted_c.view(bs, n, -1)
        
        # 处理contour特征
        contour_in_c = contour - torch.mean(contour, dim=1, keepdim=True)
        contour_in_c = contour_in_c - torch.tensor([1, 1]).cuda()
        flatted_c = torch.cat((flatted_c, contour_in_c), dim=-1)
        flatted_c = self.fc(flatted_c)
        flatted_c = flatted_c.view(-1, c_feature)
        
        # 通过GCN编码器
        l_c = self.encoder_c(flatted_c, adj)
        l_c = l_c.view(bs, n, -1)
        
        return l_c


def extract_contour_module(checkpoint_path, args, save_path):
    """
    从训练好的PairingNet模型中提取contour特征提取模块
    
    Args:
        checkpoint_path: 训练好的模型检查点路径
        args: 配置参数
        save_path: 保存独立模块的路径
    """
    from utils import pipeline
    
    # 加载完整的训练好的模型
    net = pipeline.Vanilla
    model = net(args)
    
    # 加载训练好的权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建独立的contour特征提取器
    contour_extractor = ContourFeatureExtractor(args)
    
    # 复制相关权重
    contour_extractor.flatten_net.load_state_dict(model.flatten_net.state_dict())
    contour_extractor.encoder_c.load_state_dict(model.encoder_c.state_dict())
    contour_extractor.fc.load_state_dict(model.fc.state_dict())
    
    # 保存独立的模块
    torch.save({
        'model_state_dict': contour_extractor.state_dict(),
        'args': args,
        'model_class': 'ContourFeatureExtractor'
    }, save_path)
    
    print(f"Contour特征提取模块已保存到: {save_path}")
    return contour_extractor


def load_contour_extractor(model_path):
    """
    加载保存的contour特征提取模块
    
    Args:
        model_path: 保存的模型路径
    Returns:
        contour_extractor: 加载的模型实例
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']
    
    contour_extractor = ContourFeatureExtractor(args)
    contour_extractor.load_state_dict(checkpoint['model_state_dict'])
    
    return contour_extractor, args


if __name__ == "__main__":
    # 示例使用
    print("Contour特征提取模块创建完成！")
    print("使用方法:")
    print("1. 训练完STAGE_ONE模型后，调用extract_contour_module()提取模块")
    print("2. 使用load_contour_extractor()加载保存的模块")
    print("3. 调用模块的forward()方法进行特征提取")