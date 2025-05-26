import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import spearmanr, pearsonr
import os

from model import VisionTransformer
from data import load_data, create_test_loader


def load_ensemble_models(checkpoint_dir, n_splits, device):
    """加载所有fold的最佳模型"""
    models = []
    for fold in range(1, n_splits + 1):
        model = VisionTransformer().to(device)
        model_path = os.path.join(checkpoint_dir, f"fold_{fold}_best.pth")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
            print(f"Loaded model for fold {fold}")
        else:
            print(f"Warning: Model for fold {fold} not found at {model_path}")
    
    return models


def ensemble_predict(models, test_loader, device):
    """使用模型集成进行预测"""
    ensemble_preds_list = []
    
    for i, model in enumerate(models):
        print(f"Predicting with fold {i+1} model...")
        fold_preds = []
        
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_preds.append(probs)
        
        fold_preds = np.vstack(fold_preds)
        ensemble_preds_list.append(fold_preds)
    
    # 计算平均预测
    ensemble_preds = np.mean(ensemble_preds_list, axis=0)
    return ensemble_preds


def evaluate_predictions(test_labels, ensemble_preds, results_dir):
    """评估预测结果并保存"""
    # 计算相关系数
    spearman_corr, _ = spearmanr(ensemble_preds[:, 1], test_labels)
    pearson_corr, _ = pearsonr(ensemble_preds[:, 1], test_labels)
    
    print(f'Spearman相关系数: {spearman_corr:.4f}')
    print(f'Pearson相关系数: {pearson_corr:.4f}')
    
    # 保存相关系数结果
    corr_results = pd.DataFrame({
        "Metric": ["Spearman", "Pearson"], 
        "Value": [spearman_corr, pearson_corr]
    })
    corr_results.to_csv(f"{results_dir}/ensemble_correlation.csv", index=False)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(test_labels, ensemble_preds[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of CNN-ViT')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/ensemble_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(test_labels, ensemble_preds[:, 1])
    pr_auc = auc(recall, precision)
    
    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'AP = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve of CNN-ViT')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/ensemble_pr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存所有评估指标
    metrics = {
        'Spearman': spearman_corr,
        'Pearson': pearson_corr,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc
    }
    
    # 保存详细结果
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{results_dir}/ensemble_metrics.csv", index=False)
    
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')
    
    return metrics


def test_ensemble():
    """测试模型集成"""
    # 配置参数
    from CVIT import config
    
    config = {
        'n_splits': 10,
        'batch_size': 256,
        'checkpoint_dir': str(config['CHECKPOINT_DIR']),
        'results_dir': str(config['RESULTS_DIR'])
    }
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 加载测试数据
    print("Loading test data...")
    _, _, test_data, test_labels = load_data()
    test_loader = create_test_loader(test_data, test_labels, config['batch_size'])
    
    # 加载所有模型
    print("Loading ensemble models...")
    models = load_ensemble_models(config['checkpoint_dir'], config['n_splits'], device)
    
    if len(models) == 0:
        print("Error: No models found!")
        return
    
    print(f"Loaded {len(models)} models for ensemble prediction")
    
    # 集成预测
    print("=== 开始模型集成评估 ===")
    ensemble_preds = ensemble_predict(models, test_loader, device)
    
    # 保存预测结果
    np.save(f"{config['results_dir']}/ensemble_preds.npy", ensemble_preds)
    
    # 评估预测结果
    metrics = evaluate_predictions(test_labels.numpy(), ensemble_preds, config['results_dir'])
    
    print("\n=== 最终测试结果 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return ensemble_preds, metrics


def test_single_model(model_path, test_data, test_labels, device):
    """测试单个模型"""
    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_loader = create_test_loader(test_data, test_labels, batch_size=256)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_preds.append(probs)
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.vstack(all_preds)
    return predictions, np.array(all_labels)


if __name__ == "__main__":
    ensemble_preds, metrics = test_ensemble()