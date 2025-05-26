import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np

from model import VisionTransformer
from data import load_data, create_data_loaders, get_cv_splits, check_data_split


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """保存模型检查点"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer):
    """加载模型检查点"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return model, optimizer, start_epoch, best_acc
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return model, optimizer, 0, 0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_data, batch_labels in tqdm(train_loader, desc="Training"):
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels.long())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """验证模型"""
    model.eval()
    val_correct, val_total = 0, 0
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for val_batch, val_labels in val_loader:
            val_batch, val_labels = val_batch.to(device), val_labels.to(device)
            val_outputs = model(val_batch)
            
            # 计算准确率
            _, preds = torch.max(val_outputs, dim=1)
            val_correct += (preds == val_labels.long()).sum().item()
            val_total += val_labels.size(0)
            
            # 收集用于ROC计算的数据
            probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(val_labels.cpu().numpy())
            all_probs.extend(probs)
    
    val_acc = val_correct / val_total
    return val_acc, all_labels, all_probs


def train_fold(fold, train_idx, val_idx, data, labels, device, config):
    """训练单个fold"""
    print(f"=== Fold {fold+1} 开始训练 ===")
    
    # 检查数据分割
    check_data_split(train_idx, val_idx, labels, fold)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        data, labels, train_idx, val_idx, config['batch_size']
    )
    
    # 初始化模型
    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # 训练参数
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_acc, all_val_labels, all_val_probs = validate(model, val_loader, device)
        
        print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # 保存模型权重
            torch.save(model.state_dict(), 
                      os.path.join(config['checkpoint_dir'], f"fold_{fold+1}_best.pth"))
            
            # 计算并保存ROC曲线
            fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
            roc_auc = auc(fpr, tpr)
            
            # 保存结果
            results_dir = config['results_dir']
            pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_csv(
                f'{results_dir}/fold_{fold+1}_roc.csv', index=False)
            pd.DataFrame({'Fold': [f'fold_{fold+1}'], 'AUC': [roc_auc]}).to_csv(
                f'{results_dir}/fold_{fold+1}_auc.csv', index=False)
            
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['patience']:
            print(f"Fold {fold+1} early stop at epoch {epoch+1}")
            break
    
    return best_acc


def cross_validation_train():
    """执行交叉验证训练"""
    # 配置参数
    config = {
        'n_splits': 10,
        'batch_size': 256,
        'learning_rate': 9.401718570896886e-05,
        'weight_decay': 1e-4,
        'max_epochs': 50,
        'patience': 10,
        'checkpoint_dir': "/Cas12a-CVIT/output/cv_checkpoints",
        'results_dir': "/Cas12a-CVIT/output/result"
    }
    
    # 创建目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 加载数据
    data, labels, _, _ = load_data()
    
    # 交叉验证
    cv_splits = get_cv_splits(labels, config['n_splits'])
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        best_acc = train_fold(fold, train_idx, val_idx, data, labels, device, config)
        fold_scores.append(best_acc)
        print(f"Fold {fold+1} 最佳验证准确率: {best_acc:.4f}")
    
    # 输出总体结果
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n交叉验证结果:")
    print(f"平均准确率: {mean_score:.4f} ± {std_score:.4f}")
    
    # 保存交叉验证结果
    cv_results = pd.DataFrame({
        'Fold': [f'fold_{i+1}' for i in range(len(fold_scores))],
        'Accuracy': fold_scores
    })
    cv_results.to_csv(f"{config['results_dir']}/cv_results.csv", index=False)
    
    return config


if __name__ == "__main__":
    config = cross_validation_train()