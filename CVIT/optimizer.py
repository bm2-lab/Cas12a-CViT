import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
import os
from datetime import datetime
import json
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器，用于贝叶斯优化中的目标函数"""
    
    def __init__(self, model_class, data, labels, device, config=None):
        """
        初始化模型评估器
        
        Args:
            model_class: 模型类
            data: 训练数据
            labels: 标签数据
            device: 计算设备
            config: 评估配置
        """
        self.model_class = model_class
        self.data = data
        self.labels = labels
        self.device = device
        
        # 默认配置
        self.config = {
            'batch_size': 256,
            'max_epochs': 25,
            'patience': 5,
            'validation_split': 0.2,
            'random_seed': 42,
            'use_stratified_split': True,
            'verbose': True
        }
        if config:
            self.config.update(config)
        
        # 记录日志
        self.evaluation_log = []
        self.current_trial = 0
    
    def single_fold_evaluate(self, **hyperparams) -> float:
        """
        单折交叉验证评估
        
        Args:
            **hyperparams: 超参数字典
            
        Returns:
            float: 最佳AUC分数
        """
        self.current_trial += 1
        
        # 处理超参数
        processed_params = self._process_hyperparams(hyperparams)
        
        if self.config['verbose']:
            print(f"\n=== Trial {self.current_trial} ===")
            print(f"Parameters: {processed_params}")
        
        try:
            # 数据划分
            train_loader, val_loader = self._create_data_loaders(processed_params)
            
            # 创建模型
            model = self._create_model(processed_params)
            
            # 训练和评估
            best_auc = self._train_and_evaluate(model, train_loader, val_loader, processed_params)
            
            # 记录结果
            self._log_result(processed_params, best_auc)
            
            return best_auc
            
        except Exception as e:
            print(f"Error in trial {self.current_trial}: {e}")
            return 0.0
    
    def cross_validation_evaluate(self, n_splits=5, **hyperparams) -> float:
        """
        交叉验证评估
        
        Args:
            n_splits: 交叉验证折数
            **hyperparams: 超参数字典
            
        Returns:
            float: 平均AUC分数
        """
        self.current_trial += 1
        processed_params = self._process_hyperparams(hyperparams)
        
        if self.config['verbose']:
            print(f"\n=== Trial {self.current_trial} (CV-{n_splits}) ===")
            print(f"Parameters: {processed_params}")
        
        try:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                               random_state=self.config['random_seed'])
            
            fold_aucs = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.data, self.labels)):
                if self.config['verbose']:
                    print(f"  Fold {fold+1}/{n_splits}")
                
                # 创建数据加载器
                train_loader, val_loader = self._create_fold_loaders(train_idx, val_idx)
                
                # 创建模型
                model = self._create_model(processed_params)
                
                # 训练和评估
                fold_auc = self._train_and_evaluate(model, train_loader, val_loader, processed_params)
                fold_aucs.append(fold_auc)
                
                if self.config['verbose']:
                    print(f"    Fold {fold+1} AUC: {fold_auc:.4f}")
            
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            
            if self.config['verbose']:
                print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            
            # 记录结果
            processed_params['cv_aucs'] = fold_aucs
            processed_params['cv_std'] = std_auc
            self._log_result(processed_params, mean_auc)
            
            return mean_auc
            
        except Exception as e:
            print(f"Error in trial {self.current_trial}: {e}")
            return 0.0
    
    def _process_hyperparams(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """处理超参数，确保正确的数据类型"""
        processed = {}
        
        # 整数型参数
        int_params = ['embed_dim', 'depth', 'num_heads']
        for param in int_params:
            if param in hyperparams:
                processed[param] = int(hyperparams[param])
        
        # 浮点型参数
        float_params = ['mlp_ratio', 'dropout', 'attention_drop', 'lr']
        for param in float_params:
            if param in hyperparams:
                processed[param] = float(hyperparams[param])
        
        # 添加其他参数
        for key, value in hyperparams.items():
            if key not in processed:
                processed[key] = value
        
        return processed
    
    def _create_data_loaders(self, params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器"""
        num_samples = len(self.labels)
        indices = np.arange(num_samples)
        
        if self.config['use_stratified_split']:
            # 分层采样
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                indices, test_size=self.config['validation_split'],
                stratify=self.labels, random_state=self.config['random_seed']
            )
        else:
            # 随机采样
            np.random.seed(self.config['random_seed'])
            np.random.shuffle(indices)
            split_idx = int((1 - self.config['validation_split']) * num_samples)
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        return self._create_fold_loaders(train_idx, val_idx)
    
    def _create_fold_loaders(self, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """根据索引创建数据加载器"""
        # 这里需要导入CustomDataset，假设它在model模块中
        from model import CustomDataset  # 或者从你的模块中导入
        
        train_set = CustomDataset(self.data, self.labels, train_idx)
        val_set = CustomDataset(self.data, self.labels, val_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config['batch_size'], shuffle=False)
        
        return train_loader, val_loader
    
    def _create_model(self, params: Dict[str, Any]):
        """创建模型实例"""
        model_params = {
            'embed_dim': params.get('embed_dim', 256),
            'depth': params.get('depth', 6),
            'num_heads': params.get('num_heads', 8),
            'mlp_ratio': params.get('mlp_ratio', 2.0),
            'dropout': params.get('dropout', 0.3),
            'attention_drop': params.get('attention_drop', 0.3)
        }
        
        return self.model_class(**model_params).to(self.device)
    
    def _train_and_evaluate(self, model, train_loader: DataLoader, 
                           val_loader: DataLoader, params: Dict[str, Any]) -> float:
        """训练和评估模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 1e-4))
        
        best_auc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            # 训练
            model.train()
            train_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels.long())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            val_auc = self._validate_model(model, val_loader)
            
            if self.config['verbose']:
                avg_train_loss = train_loss / len(train_loader)
                print(f"    Epoch {epoch+1:02d} - Loss: {avg_train_loss:.4f}, AUC: {val_auc:.4f}")
            
            # 早停检查
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                if self.config['verbose']:
                    print(f"    Early stopping at epoch {epoch+1}. Best AUC: {best_auc:.4f}")
                break
        
        return best_auc
    
    def _validate_model(self, model, val_loader: DataLoader) -> float:
        """验证模型性能"""
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                outputs = model(val_data)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                all_probs.extend(probs)
                all_labels.extend(val_labels.cpu().numpy())
        
        return roc_auc_score(all_labels, all_probs)
    
    def _log_result(self, params: Dict[str, Any], auc: float):
        """记录评估结果"""
        log_entry = {
            'trial': self.current_trial,
            'timestamp': datetime.now().isoformat(),
            'auc': auc,
            **params
        }
        self.evaluation_log.append(log_entry)
    
    def get_evaluation_log(self) -> pd.DataFrame:
        """获取评估日志"""
        return pd.DataFrame(self.evaluation_log)


class BayesianHyperparameterOptimizer:
    """贝叶斯超参数优化器"""
    
    def __init__(self, model_class, data, labels, device, output_dir=os.path.join(config['RESULTS_DIR'], "bayes_opt")):
        """
        初始化优化器
        
        Args:
            model_class: 模型类
            data: 训练数据
            labels: 标签数据  
            device: 计算设备
            output_dir: 输出目录
        """
        self.model_class = model_class
        self.data = data
        self.labels = labels
        self.device = device
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        
        # 初始化评估器
        self.evaluator = None
        self.optimizer = None
        
    def setup_search_space(self, custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Tuple[float, float]]:
        """设置搜索空间"""
        default_bounds = {
            'embed_dim': (64, 512),
            'depth': (4, 12),
            'num_heads': (4, 12),
            'mlp_ratio': (1.5, 4.5),
            'dropout': (0.1, 0.5),
            'attention_drop': (0.1, 0.5),
            'lr': (1e-5, 5e-3)
        }
        
        if custom_bounds:
            default_bounds.update(custom_bounds)
        
        return default_bounds
    
    def optimize(self, 
                search_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                init_points: int = 5,
                n_iter: int = 15,
                evaluation_method: str = 'single_fold',
                n_splits: int = 5,
                evaluator_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行贝叶斯优化
        
        Args:
            search_bounds: 搜索边界
            init_points: 初始探索点数
            n_iter: 优化迭代次数
            evaluation_method: 评估方法 ('single_fold' 或 'cross_validation')
            n_splits: 交叉验证折数（仅当evaluation_method='cross_validation'时使用）
            evaluator_config: 评估器配置
            
        Returns:
            Dict: 优化结果
        """
        print(f"Starting Bayesian Optimization...")
        print(f"Evaluation method: {evaluation_method}")
        print(f"Device: {self.device}")
        
        # 设置搜索空间
        pbounds = self.setup_search_space(search_bounds)
        print(f"Search bounds: {pbounds}")
        
        # 初始化评估器
        self.evaluator = ModelEvaluator(
            self.model_class, self.data, self.labels, self.device, evaluator_config
        )
        
        # 选择评估函数
        if evaluation_method == 'single_fold':
            objective_function = self.evaluator.single_fold_evaluate
        elif evaluation_method == 'cross_validation':
            objective_function = lambda **params: self.evaluator.cross_validation_evaluate(n_splits, **params)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")
        
        # 初始化贝叶斯优化器
        self.optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        # 执行优化
        print(f"\nStarting optimization with {init_points} initial points and {n_iter} iterations...")
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # 获取最优结果
        best_params = self.optimizer.max
        print(f"\nOptimization completed!")
        print(f"Best parameters: {best_params}")
        
        # 保存结果
        self._save_results(evaluation_method, n_splits)
        
        return best_params
    
    def _save_results(self, evaluation_method: str, n_splits: int):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存评估日志
        if self.evaluator:
            log_df = self.evaluator.get_evaluation_log()
            log_filename = f"hyperparameter_log_{evaluation_method}_{timestamp}.csv"
            log_path = os.path.join(self.output_dir, "logs", log_filename)
            log_df.to_csv(log_path, index=False)
            print(f"Evaluation log saved to: {log_path}")
        
        # 保存最优参数
        if self.optimizer:
            best_params = self.optimizer.max
            results = {
                'timestamp': timestamp,
                'evaluation_method': evaluation_method,
                'n_splits': n_splits if evaluation_method == 'cross_validation' else 1,
                'best_score': best_params['target'],
                'best_params': best_params['params'],
                'total_trials': len(self.optimizer.space.keys)
            }
            
            results_filename = f"best_params_{evaluation_method}_{timestamp}.json"
            results_path = os.path.join(self.output_dir, "results", results_filename)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Best parameters saved to: {results_path}")
    
    def load_and_continue(self, log_file: str, additional_iter: int = 10):
        """从之前的日志继续优化"""
        # 这个功能可以后续实现，用于从中断的优化中继续
        pass


def main():
    """主函数示例"""
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    from CVIT import config
    data_path = os.path.join(config['DATA_DIR'], "dataset1.pt")
    labels_path = os.path.join(config['DATA_DIR'], "labelset1.pt")
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Data files not found in {config['DATA_DIR']}")
    
    data = torch.load(data_path)
    labels = torch.load(labels_path)
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 导入模型类
    from model import VisionTransformer
    
    # 创建优化器
    optimizer = BayesianHyperparameterOptimizer(
        model_class=VisionTransformer,
        data=data,
        labels=labels,
        device=device,
        output_dir=os.path.join(config['RESULTS_DIR'], "bayes_opt")
    )
    
    # 自定义评估器配置
    evaluator_config = {
        'batch_size': 256,
        'max_epochs': 25,
        'patience': 5,
        'validation_split': 0.2,
        'verbose': True
    }
    
    # 执行单折优化
    print("\n=== Single Fold Optimization ===")
    best_params_single = optimizer.optimize(
        init_points=5,
        n_iter=10,
        evaluation_method='single_fold',
        evaluator_config=evaluator_config
    )
    
    # 执行交叉验证优化（可选）
    # print("\n=== Cross Validation Optimization ===")
    # best_params_cv = optimizer.optimize(
    #     init_points=3,
    #     n_iter=7,
    #     evaluation_method='cross_validation',
    #     n_splits=3,
    #     evaluator_config=evaluator_config
    # )

if __name__ == "__main__":
    main()