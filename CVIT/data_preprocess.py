import torch
import numpy as np
import pandas as pd
from functools import reduce
from operator import add
import os


class SequenceProcessor:
    """序列数据处理器"""
    
    # 核苷酸映射
    NUCLEOTIDE_MAP = {
        'A': (1, 0, 0, 0),
        'C': (0, 1, 0, 0),
        'G': (0, 0, 1, 0),
        'T': (0, 0, 0, 1)
    }
    
    # 表观遗传学特征映射
    EPI_MAP = {'A': 1, 'N': 0}
    
    @classmethod
    def get_sequence_code(cls, seq):
        """将DNA序列转换为one-hot编码"""
        try:
            encoded = reduce(add, map(lambda c: cls.NUCLEOTIDE_MAP[c], seq.upper()))
            return np.array(encoded).reshape((1, len(seq), -1))
        except KeyError as e:
            raise ValueError(f"Invalid nucleotide '{e.args[0]}' found in sequence: {seq}")
    
    @classmethod
    def get_epi_code(cls, eseq):
        """将表观遗传学序列转换为编码"""
        try:
            encoded = list(map(lambda c: cls.EPI_MAP[c], eseq))
            return np.array(encoded).reshape(1, len(eseq), -1)
        except KeyError as e:
            raise ValueError(f"Invalid epigenetic marker '{e.args[0]}' found in sequence: {eseq}")


class EpiSGT:
    """表观遗传学序列-基因型数据集处理器"""
    
    def __init__(self, file_path, num_epi_features, with_labels=True):
        """
        初始化数据集处理器
        
        Args:
            file_path (str): 数据文件路径
            num_epi_features (int): 表观遗传学特征数量
            with_labels (bool): 是否包含标签
        """
        self.file_path = file_path
        self.num_epi_features = num_epi_features
        self.with_labels = with_labels
        
        # 加载和验证数据
        self._load_data()
        self._validate_data()
    
    def _load_data(self):
        """加载数据文件"""
        try:
            self.original_df = pd.read_csv(self.file_path, delim_whitespace=True, header=None)
            print(f"Successfully loaded data from: {self.file_path}")
            print(f"Data shape: {self.original_df.shape}")
            print(f"Columns: {list(self.original_df.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"Error loading file {self.file_path}: {e}")
    
    def _validate_data(self):
        """验证数据格式和完整性"""
        expected_cols = self.num_epi_features + 2 if self.with_labels else self.num_epi_features + 1
        actual_cols = len(self.original_df.columns)
        
        if actual_cols < expected_cols:
            raise ValueError(
                f"Insufficient columns in data file. "
                f"Expected: {expected_cols}, Found: {actual_cols}. "
                f"Check file content or adjust num_epi_features parameter."
            )
        
        # 选择需要的列（从末尾开始选择）
        self.selected_columns = list(self.original_df.columns)[-expected_cols:]
        self.data_df = self.original_df[self.selected_columns]
        
        print(f"Selected columns: {self.selected_columns}")
        print(f"Sample data:\n{self.data_df.head()}")
    
    @property
    def length(self):
        """返回数据集长度"""
        return len(self.data_df)
    
    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        """
        获取处理后的数据集
        
        Args:
            x_dtype: 特征数据类型
            y_dtype: 标签数据类型
            
        Returns:
            tuple: (X, y) 如果with_labels=True，否则只返回 X
        """
        print("Processing sequences...")
        
        # 处理DNA序列
        sequences = self.data_df[self.selected_columns[0]]
        x_seq = np.concatenate([SequenceProcessor.get_sequence_code(seq) for seq in sequences])
        
        # 处理表观遗传学特征
        epi_features = []
        for i in range(1, 1 + self.num_epi_features):
            col_name = self.selected_columns[i]
            epi_seqs = self.data_df[col_name]
            epi_encoded = np.concatenate([SequenceProcessor.get_epi_code(eseq) for eseq in epi_seqs])
            epi_features.append(epi_encoded)
        
        # 合并表观遗传学特征
        if epi_features:
            x_epis = np.concatenate(epi_features, axis=-1)
            # 合并序列和表观遗传学特征
            x = np.concatenate([x_seq, x_epis], axis=-1).astype(x_dtype)
        else:
            x = x_seq.astype(x_dtype)
        
        # 转置以符合模型输入格式
        x = x.transpose(0, 2, 1)
        
        print(f"Processed data shape: {x.shape}")
        
        if self.with_labels:
            y = np.array(self.data_df[self.selected_columns[-1]]).astype(y_dtype)
            print(f"Labels shape: {y.shape}")
            print(f"Label distribution: {np.bincount(y.astype(int))}")
            return x, y
        else:
            return x


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, output_dir=None):
        from CVIT import config
        
        if output_dir is None:
            output_dir = str(config['DATA_DIR'])
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        """
        初始化数据集构建器
        
        Args:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_and_save(self, file_path, num_epi_features, output_prefix, 
                        target_shape=None, normalize=False, standardize=False):
        """
        处理数据并保存
        
        Args:
            file_path (str): 输入文件路径
            num_epi_features (int): 表观遗传学特征数量
            output_prefix (str): 输出文件前缀
            target_shape (tuple): 目标形状，例如 (1, 6, 34)
            normalize (bool): 是否归一化
            standardize (bool): 是否标准化
        """
        print(f"Processing file: {file_path}")
        
        # 创建数据处理器
        processor = EpiSGT(file_path, num_epi_features, with_labels=True)
        
        # 获取数据集
        x, y = processor.get_dataset()
        
        # 重塑数据
        if target_shape:
            x = x.reshape([-1] + list(target_shape))
            print(f"Reshaped data to: {x.shape}")
        
        # 数据预处理
        if normalize:
            x = self._normalize_data(x)
            print("Applied normalization")
        
        if standardize:
            x = self._standardize_data(x)
            print("Applied standardization")
        
        # 保存数据
        data_path = os.path.join(self.output_dir, f"{output_prefix}.pt")
        label_path = os.path.join(self.output_dir, f"labelset{output_prefix[7:]}.pt")
        
        torch.save(x, data_path)
        torch.save(y, label_path)
        
        print(f"Saved processed data to: {data_path}")
        print(f"Saved labels to: {label_path}")
        print(f"Final data shape: {x.shape}")
        print(f"Final labels shape: {y.shape}")
        
        return x, y
    
    def _normalize_data(self, data):
        """归一化数据到 [0, 1] 范围"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return data
    
    def _standardize_data(self, data):
        """标准化数据 (z-score)"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            return (data - mean_val) / std_val
        return data


def main():
    """主函数，演示如何使用数据处理器"""
    
    # 配置参数
    config = {
        'input_file': os.path.join(config['DATA_DIR_PROCESSED'], "Dataset_HT_1_1.txt"),
        'num_epi_features': 2,
        'output_dir': config['DATA_DIR'],
        'output_prefix': "dataset1",
        'target_shape': (1, 6, 34),
        'normalize': False,
        'standardize': False
    }
    
    # 创建数据集构建器
    builder = DatasetBuilder(config['output_dir'])
    
    # 处理并保存数据
    try:
        x, y = builder.process_and_save(
            file_path=config['input_file'],
            num_epi_features=config['num_epi_features'],
            output_prefix=config['output_prefix'],
            target_shape=config['target_shape'],
            normalize=config['normalize'],
            standardize=config['standardize']
        )
        
        print("\n=== Processing Summary ===")
        print(f"Successfully processed {len(y)} samples")
        print(f"Feature dimensions: {x.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()