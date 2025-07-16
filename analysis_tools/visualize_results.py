import matplotlib.pyplot as plt
import json
import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from network_improved import EdgeSparkNet
from dataset_loader import create_dataloaders

def plot_training_history(history_path, save_dir):
    """绘制训练历史"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['epoch'], history['train_acc'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['epoch'], history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1分数曲线
    axes[1, 0].plot(history['epoch'], history['val_f1'], label='验证F1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1分数曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC曲线
    axes[1, 1].plot(history['epoch'], history['val_auc'], label='验证AUC', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('AUC曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练历史图保存到: {os.path.join(save_dir, 'training_history.png')}")

def evaluate_model(model_path, test_loader, device):
    """评估模型"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = EdgeSparkNet(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("评估模型...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            source_points = batch['source_points'].to(device)
            target_points = batch['target_points'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(source_points, target_points)
            
            all_predictions.extend((predictions > 0.5).cpu().numpy())
            all_probabilities.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_loader)}")
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['不匹配', '匹配'], 
                yticklabels=['不匹配', '匹配'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵保存到: {os.path.join(save_dir, 'confusion_matrix.png')}")

def plot_probability_distribution(y_true, y_prob, save_dir):
    """绘制概率分布"""
    plt.figure(figsize=(12, 5))
    
    # 分别绘制正样本和负样本的概率分布
    plt.subplot(1, 2, 1)
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    plt.hist(neg_probs, bins=50, alpha=0.7, label='负样本', color='red', density=True)
    plt.hist(pos_probs, bins=50, alpha=0.7, label='正样本', color='blue', density=True)
    plt.xlabel('预测概率')
    plt.ylabel('密度')
    plt.title('预测概率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制ROC曲线样式的概率分布
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        pred_binary = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (pred_binary == 1))
        fp = np.sum((y_true == 0) & (pred_binary == 1))
        tn = np.sum((y_true == 0) & (pred_binary == 0))
        fn = np.sum((y_true == 1) & (pred_binary == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    plt.plot(fpr_list, tpr_list, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"概率分析图保存到: {os.path.join(save_dir, 'probability_analysis.png')}")

def generate_report(exp_dir):
    """生成完整的实验报告"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查必要文件
    model_path = os.path.join(exp_dir, 'best_model.pth')
    history_path = os.path.join(exp_dir, 'training_history.json')
    config_path = os.path.join(exp_dir, 'config.json')
    
    if not all(os.path.exists(p) for p in [model_path, history_path, config_path]):
        print("缺少必要的文件，无法生成报告")
        return
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"为实验生成报告: {exp_dir}")
    
    # 1. 绘制训练历史
    plot_training_history(history_path, exp_dir)
    
    # 2. 创建测试数据加载器
    print("创建测试数据加载器...")
    _, _, test_loader = create_dataloaders(
        "dataset/train_set.pkl",  # 这里只是为了创建加载器，实际只会用test
        "dataset/valid_set.pkl",
        "dataset/test_set.pkl",
        batch_size=config['data']['batch_size'],
        max_points=config['data']['max_points'],
        num_workers=0  # 评估时不使用多进程
    )
    
    # 3. 评估模型
    y_true, y_pred, y_prob = evaluate_model(model_path, test_loader, device)
    
    # 4. 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, exp_dir)
    
    # 5. 绘制概率分析
    plot_probability_distribution(y_true, y_prob, exp_dir)
    
    # 6. 生成文本报告
    report = classification_report(y_true, y_pred, target_names=['不匹配', '匹配'])
    
    with open(os.path.join(exp_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("EdgeSpark模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"实验配置:\n")
        f.write(json.dumps(config, indent=2, ensure_ascii=False))
        f.write("\n\n")
        f.write("测试集评估结果:\n")
        f.write(report)
        f.write(f"\n\n总体准确率: {np.mean(y_true == y_pred):.4f}")
        f.write(f"\n正样本数量: {np.sum(y_true == 1)}")
        f.write(f"\n负样本数量: {np.sum(y_true == 0)}")
    
    print(f"完整报告生成完毕，保存在: {exp_dir}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python visualize_results.py <experiment_directory>")
        print("例如: python visualize_results.py experiments/exp_20231201_120000")
        return
    
    exp_dir = sys.argv[1]
    if not os.path.exists(exp_dir):
        print(f"实验目录不存在: {exp_dir}")
        return
    
    generate_report(exp_dir)

if __name__ == "__main__":
    # 如果没有命令行参数，列出可用的实验
    import sys
    if len(sys.argv) == 1:
        print("可用的实验目录:")
        if os.path.exists('experiments'):
            for exp in os.listdir('experiments'):
                if exp.startswith('exp_'):
                    print(f"  {os.path.join('experiments', exp)}")
        else:
            print("  没有找到experiments目录")
        print("\n使用方法: python visualize_results.py <experiment_directory>")
    else:
        main()