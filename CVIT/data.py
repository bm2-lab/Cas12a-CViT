import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data, labels, indices):
        self.data = data[indices]  # 预先筛选索引
        self.labels = labels[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def load_data():
    """加载训练和测试数据"""
    # 加载数据
    data = torch.load("/root/project/data/dataset1.pt")
    labels = torch.load("/root/project/data/labelset1.pt")
    test_data = torch.load("/root/project/data/dataset2.pt")
    test_labels = torch.load("/root/project/data/labelset2.pt")

    # 转换数据类型
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    return data, labels, test_data, test_labels


def create_data_loaders(data, labels, train_idx, val_idx, batch_size=256):
    """创建训练和验证数据加载器"""
    train_dataset = CustomDataset(data, labels, train_idx)
    val_dataset = CustomDataset(data, labels, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_test_loader(test_data, test_labels, batch_size=256):
    """创建测试数据加载器"""
    test_dataset = CustomDataset(test_data, test_labels, list(range(len(test_data))))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_cv_splits(labels, n_splits=10, random_state=42):
    """获取交叉验证分割"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kf.split(np.zeros(len(labels)), labels)


def check_data_split(train_idx, val_idx, labels, fold):
    """检查数据分割情况"""
    overlap = len(set(train_idx) & set(val_idx))
    print(f"Fold {fold+1}: Train/Val 重叠样本数: {overlap}")
    unique, counts = np.unique(labels.numpy().astype(int), return_counts=True)
    print(f"Fold {fold+1} 标签分布: {dict(zip(unique, counts))}")