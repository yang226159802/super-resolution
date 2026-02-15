import numpy as np
import torch
from torch.utils.data import Dataset

# flame_dataset.py
class FlameDataset(Dataset):
    def __init__(self, npz_path, normalize=True, norm_method='zscore', stats_dict=None):
        """
        Args:
            npz_path: npz文件路径
            normalize: 是否归一化
            norm_method: 归一化方法，可选 'zscore', 'minmax', 'none'
            stats_dict: 外部传入的统计信息字典（用于统一归一化）
        """
        data = np.load(npz_path)
        self.lr_data = data['lr'].astype(np.float32)
        self.hr_data = data['hr'].astype(np.float32)
        
        print(f"加载数据集: {npz_path}")
        print(f"LR形状: {self.lr_data.shape}, HR形状: {self.hr_data.shape}")
        print(f"归一化方法: {norm_method}")
        
        self.norm_method = norm_method
        self.normalize = normalize
        
        if normalize:
            if stats_dict is not None:
                # 使用外部传入的统计信息
                self.load_external_stats(stats_dict)
            else:
                # 自己计算统计信息（训练集）
                self.normalize_data()
    
    def normalize_data(self):
        """根据选择的归一化方法归一化数据"""
        if self.norm_method == 'zscore':
            # Z-score归一化每个通道
            self.lr_mean = self.lr_data.mean(axis=(0, 2, 3), keepdims=True)
            self.lr_std = self.lr_data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            
            self.hr_mean = self.hr_data.mean(axis=(0, 2, 3), keepdims=True)
            self.hr_std = self.hr_data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            
            # 归一化
            self.lr_data = (self.lr_data - self.lr_mean) / self.lr_std
            self.hr_data = (self.hr_data - self.hr_mean) / self.hr_std
            
            # 保存统计信息
            self.norm_stats = {
                'method': 'zscore',
                'lr_mean': self.lr_mean,
                'lr_std': self.lr_std,
                'hr_mean': self.hr_mean,
                'hr_std': self.hr_std
            }
            
        elif self.norm_method == 'minmax':
            # Min-Max归一化到[0, 1]
            self.lr_min = self.lr_data.min(axis=(0, 2, 3), keepdims=True)
            self.lr_max = self.lr_data.max(axis=(0, 2, 3), keepdims=True)
            
            self.hr_min = self.hr_data.min(axis=(0, 2, 3), keepdims=True)
            self.hr_max = self.hr_data.max(axis=(0, 2, 3), keepdims=True)
            
            # 归一化
            self.lr_data = (self.lr_data - self.lr_min) / (self.lr_max - self.lr_min + 1e-8)
            self.hr_data = (self.hr_data - self.hr_min) / (self.hr_max - self.hr_min + 1e-8)
            
            # 保存统计信息
            self.norm_stats = {
                'method': 'minmax',
                'lr_min': self.lr_min,
                'lr_max': self.lr_max,
                'hr_min': self.hr_min,
                'hr_max': self.hr_max
            }
            
        elif self.norm_method == 'none' or self.norm_method is None:
            # 不进行归一化
            self.norm_stats = {'method': 'none'}
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.norm_method}")
    
    def load_external_stats(self, stats_dict):
        """加载外部统计信息"""
        self.norm_stats = stats_dict
        
        if stats_dict['method'] == 'zscore':
            self.lr_mean = stats_dict['lr_mean']
            self.lr_std = stats_dict['lr_std']
            self.hr_mean = stats_dict['hr_mean']
            self.hr_std = stats_dict['hr_std']
            
            self.lr_data = (self.lr_data - self.lr_mean) / self.lr_std
            self.hr_data = (self.hr_data - self.hr_mean) / self.hr_std
            
        elif stats_dict['method'] == 'minmax':
            self.lr_min = stats_dict['lr_min']
            self.lr_max = stats_dict['lr_max']
            self.hr_min = stats_dict['hr_min']
            self.hr_max = stats_dict['hr_max']
            
            self.lr_data = (self.lr_data - self.lr_min) / (self.lr_max - self.lr_min + 1e-8)
            self.hr_data = (self.hr_data - self.hr_min) / (self.hr_max - self.hr_min + 1e-8)
            
        elif stats_dict['method'] == 'none':
            pass  # 不进行归一化
    
    def denormalize(self, data, is_lr=True):
        """反归一化"""
        if not self.normalize or self.norm_stats['method'] == 'none':
            return data
        
        if self.norm_stats['method'] == 'zscore':
            if is_lr:
                if hasattr(self, 'lr_mean') and hasattr(self, 'lr_std'):
                    return data * self.lr_std + self.lr_mean
            else:
                if hasattr(self, 'hr_mean') and hasattr(self, 'hr_std'):
                    return data * self.hr_std + self.hr_mean
        
        elif self.norm_stats['method'] == 'minmax':
            if is_lr:
                if hasattr(self, 'lr_min') and hasattr(self, 'lr_max'):
                    return data * (self.lr_max - self.lr_min) + self.lr_min
            else:
                if hasattr(self, 'hr_min') and hasattr(self, 'hr_max'):
                    return data * (self.hr_max - self.hr_min) + self.hr_min
        
        # 如果没有对应的统计信息，返回原始数据
        return data
    
    def get_stats_dict(self):
        """获取归一化统计信息字典"""
        if not self.normalize:
            return {'method': 'none'}
        return self.norm_stats
    
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_data[idx])
        hr = torch.from_numpy(self.hr_data[idx])
        return lr, hr