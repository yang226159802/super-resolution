# evaluate_flame_sr.py
# -*- coding: utf-8 -*-
"""
火焰超分辨率模型评估脚本
功能：
1. 加载训练好的模型
2. 加载测试集（与训练集相同格式）
3. 计算评估指标（PSNR, SSIM, RMSE等）
4. 可视化对比结果
5. 保存评估结果
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import ndimage
from tqdm import tqdm
import json
import pickle
from pathlib import Path

#绘图显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和损失函数
from flame_edsr2d import FlameEDSR2D

print(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.cuda.set_device(0)


# # 设置matplotlib中文字体和样式
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")

def plot_channel_metrics(test_dataset, evaluator, overall_metrics, config):
    """
    绘制各通道指标对比图
    """
    # 获取测试集中的一个样本用于分析各通道
    sample_idx = min(1, len(test_dataset) - 1)
    lr_sample, hr_sample = test_dataset.get_sample(sample_idx)
    
    # 模型推理
    with torch.no_grad():
        lr_tensor = lr_sample.unsqueeze(0).to(config['device'])
        sr_tensor = evaluator.model(lr_tensor)
        sr_sample = sr_tensor.cpu().numpy()[0]
    
    # 计算各通道指标
    n_channels = min(sr_sample.shape[0], len(evaluator.channel_names))
    channel_metrics = {
        'psnr': [],
        'ssim': [],
        'rmse': [],
        'mae': []
    }
    
    for ch_idx in range(n_channels):
        ch_name = evaluator.channel_names[ch_idx]
        sr_ch = sr_sample[ch_idx]
        hr_ch = hr_sample[ch_idx].numpy()
        
        # 计算指标
        channel_metrics['psnr'].append(evaluator.calculate_psnr(sr_ch, hr_ch))
        channel_metrics['ssim'].append(evaluator.calculate_ssim(sr_ch, hr_ch))
        channel_metrics['rmse'].append(np.sqrt(np.mean((sr_ch - hr_ch) ** 2)))
        channel_metrics['mae'].append(np.mean(np.abs(sr_ch - hr_ch)))
    
    # 创建条形图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metric_names = ['psnr', 'ssim', 'rmse', 'mae']
    metric_titles = ['PSNR (dB)', 'SSIM', 'RMSE', 'MAE']
    
    for i, (metric, title) in enumerate(zip(metric_names, metric_titles)):
        ax = axes[i]
        x_pos = np.arange(n_channels)
        bars = ax.bar(x_pos, channel_metrics[metric], alpha=0.7, color='steelblue')
        
        ax.set_xlabel('通道')
        ax.set_ylabel(title)
        ax.set_title(f'各通道{title}对比')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(evaluator.channel_names[:n_channels], rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, channel_metrics[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle('各通道评估指标对比', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(config['output_dir'], 'channel_metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存通道指标对比图到: {save_path}")
    plt.close()

#FlameTestDataset 类
class FlameTestDataset:
    """
    火焰测试数据集类
    """
    def __init__(self, test_npz_path, normalization_stats_path=None, scale=8, 
                 norm_method='zscore', stats_dict=None):
        """
        初始化测试数据集
        
        Args:
            test_npz_path: 测试集npz文件路径
            normalization_stats_path: 归一化统计信息文件路径
            scale: 超分比例
            norm_method: 归一化方法，可选 'zscore', 'minmax', 'none'
            stats_dict: 归一化统计信息字典
        """
        self.scale = scale
        self.test_npz_path = test_npz_path
        self.norm_method = norm_method
        
        # 加载测试数据
        print(f"加载测试集: {test_npz_path}")
        data = np.load(test_npz_path)
        self.lr_data = data['lr'].astype(np.float32)
        self.hr_data = data['hr'].astype(np.float32)
        
        print(f"测试集尺寸: LR={self.lr_data.shape}, HR={self.hr_data.shape}")
        print(f"归一化方法: {norm_method}")
        
        # 验证尺寸关系
        lr_h, lr_w = self.lr_data.shape[2], self.lr_data.shape[3]
        hr_h, hr_w = self.hr_data.shape[2], self.hr_data.shape[3]
        
        if hr_h != lr_h * scale or hr_w != lr_w * scale:
            print(f"警告: HR尺寸({hr_h}, {hr_w}) != LR尺寸({lr_h}, {lr_w}) × {scale}")
            print("正在调整HR尺寸以匹配...")
            self.adjust_hr_size()
        
        # 加载归一化统计信息
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, 'rb') as f:
                self.normalization_stats = pickle.load(f)
            print("已加载归一化统计信息文件")
        else:
            self.normalization_stats = None
        
        # 优先使用传入的stats_dict，否则使用从文件加载的统计信息
        if stats_dict is not None:
            self.stats_dict = stats_dict
        elif self.normalization_stats is not None:
            self.stats_dict = self.normalization_stats
        else:
            print("警告: 未找到归一化统计信息，将使用测试集自身的统计信息")
            self.stats_dict = None
        
        # 应用归一化
        self.apply_normalization()
    
    def adjust_hr_size(self):
        """调整HR尺寸以匹配LR×scale"""
        lr_h, lr_w = self.lr_data.shape[2], self.lr_data.shape[3]
        target_h = lr_h * self.scale
        target_w = lr_w * self.scale
        
        adjusted_hr = []
        for i in range(self.hr_data.shape[0]):
            sample = self.hr_data[i]
            adjusted_channels = []
            for c in range(sample.shape[0]):
                channel_data = sample[c]
                adjusted_channel = ndimage.zoom(
                    channel_data,
                    (target_h / channel_data.shape[0], target_w / channel_data.shape[1]),
                    order=1
                )
                adjusted_channels.append(adjusted_channel)
            adjusted_hr.append(np.stack(adjusted_channels, axis=0))
        
        self.hr_data = np.array(adjusted_hr).astype(np.float32)
        print(f"调整后HR形状: {self.hr_data.shape}")
    
    def apply_normalization(self):
        """根据选择的归一化方法应用归一化"""
        if self.norm_method == 'none' or self.norm_method is None:
            print("不进行归一化")
            self.norm_stats = {'method': 'none'}
            return
        
        if self.stats_dict and 'method' in self.stats_dict:
            # 使用外部统计信息
            stats = self.stats_dict
            if stats['method'] == 'zscore':
                # Z-score归一化
                if 'lr_mean' in stats and 'lr_std' in stats:
                    self.lr_mean = stats['lr_mean']
                    self.lr_std = stats['lr_std']
                    self.hr_mean = stats['hr_mean']
                    self.hr_std = stats['hr_std']
                    
                    self.lr_data = (self.lr_data - self.lr_mean) / self.lr_std
                    self.hr_data = (self.hr_data - self.hr_mean) / self.hr_std
                    
                    self.norm_stats = {
                        'method': 'zscore',
                        'lr_mean': self.lr_mean,
                        'lr_std': self.lr_std,
                        'hr_mean': self.hr_mean,
                        'hr_std': self.hr_std
                    }
                else:
                    print("警告: Z-score统计信息不完整，使用测试集自身统计信息")
                    self.compute_zscore_stats()
                    
            elif stats['method'] == 'minmax':
                # Min-Max归一化
                if 'lr_min' in stats and 'lr_max' in stats:
                    self.lr_min = stats['lr_min']
                    self.lr_max = stats['lr_max']
                    self.hr_min = stats['hr_min']
                    self.hr_max = stats['hr_max']
                    
                    self.lr_data = (self.lr_data - self.lr_min) / (self.lr_max - self.lr_min + 1e-8)
                    self.hr_data = (self.hr_data - self.hr_min) / (self.hr_max - self.hr_min + 1e-8)
                    
                    self.norm_stats = {
                        'method': 'minmax',
                        'lr_min': self.lr_min,
                        'lr_max': self.lr_max,
                        'hr_min': self.hr_min,
                        'hr_max': self.hr_max
                    }
                else:
                    print("警告: Min-Max统计信息不完整，使用测试集自身统计信息")
                    self.compute_minmax_stats()
            else:
                print(f"警告: 不支持的归一化方法: {stats['method']}，使用测试集自身统计信息")
                self.compute_zscore_stats()
        else:
            # 使用测试集自身统计信息
            print("使用测试集自身统计信息")
            if self.norm_method == 'zscore':
                self.compute_zscore_stats()
            elif self.norm_method == 'minmax':
                self.compute_minmax_stats()
    
    def compute_zscore_stats(self):
        """计算Z-score归一化统计信息"""
        self.lr_mean = self.lr_data.mean(axis=(0, 2, 3), keepdims=True)
        self.lr_std = self.lr_data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        self.hr_mean = self.hr_data.mean(axis=(0, 2, 3), keepdims=True)
        self.hr_std = self.hr_data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        
        # 应用归一化
        self.lr_data = (self.lr_data - self.lr_mean) / self.lr_std
        self.hr_data = (self.hr_data - self.hr_mean) / self.hr_std
        
        self.norm_stats = {
            'method': 'zscore',
            'lr_mean': self.lr_mean,
            'lr_std': self.lr_std,
            'hr_mean': self.hr_mean,
            'hr_std': self.hr_std
        }
    
    def compute_minmax_stats(self):
        """计算Min-Max归一化统计信息"""
        self.lr_min = self.lr_data.min(axis=(0, 2, 3), keepdims=True)
        self.lr_max = self.lr_data.max(axis=(0, 2, 3), keepdims=True)
        self.hr_min = self.hr_data.min(axis=(0, 2, 3), keepdims=True)
        self.hr_max = self.hr_data.max(axis=(0, 2, 3), keepdims=True)
        
        # 应用归一化
        self.lr_data = (self.lr_data - self.lr_min) / (self.lr_max - self.lr_min + 1e-8)
        self.hr_data = (self.hr_data - self.hr_min) / (self.hr_max - self.hr_min + 1e-8)
        
        self.norm_stats = {
            'method': 'minmax',
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'hr_min': self.hr_min,
            'hr_max': self.hr_max
        }
    
    def denormalize(self, data, is_lr=True):
        """
        反归一化
        
        Args:
            data: 需要反归一化的数据
            is_lr: 是否是LR数据
            
        Returns:
            反归一化后的数据
        """
        if not hasattr(self, 'norm_stats') or self.norm_stats['method'] == 'none':
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
    
    def get_sample(self, idx):
        """获取指定索引的样本"""
        lr = torch.from_numpy(self.lr_data[idx])
        hr = torch.from_numpy(self.hr_data[idx])
        return lr, hr
    
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        return self.get_sample(idx)

class FlameEvaluator:
    """
    火焰超分辨率模型评估器
    """
    def __init__(self, model, device='cuda'):
        """
        初始化评估器
        
        Args:
            model: 训练好的模型
            device: 设备（cuda或cpu）
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # 评估指标
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'rmse': [],
            'mae': [],
            'grad_loss': []
        }
        
        # 损失函数
        # self.grad_loss_fn = GradientLoss(norm="l1", reduce="mean").to(device)
        
        # 通道名称（根据模型输入输出通道数）
        self.channel_names = [
            'T', 'H2', 'O2', 'H2O', 'u', 'v', 'w', 'p', 'rho'
        ]
        
        # 确保通道名称数量与模型通道数匹配
        if hasattr(model, 'in_channels'):
            actual_channels = model.in_channels
            if len(self.channel_names) > actual_channels:
                self.channel_names = self.channel_names[:actual_channels]
            elif len(self.channel_names) < actual_channels:
                # 添加默认名称
                for i in range(len(self.channel_names), actual_channels):
                    self.channel_names.append(f'Channel_{i}')
        
        print(f"模型输入通道数: {model.in_channels if hasattr(model, 'in_channels') else '未知'}")
        print(f"模型输出通道数: {model.out_channels if hasattr(model, 'out_channels') else '未知'}")
        print(f"通道名称: {self.channel_names}")
    
    def calculate_psnr(self, img1, img2, max_val=1.0):
        """
        计算峰值信噪比
        
        Args:
            img1: 图像1
            img2: 图像2
            max_val: 最大像素值
            
        Returns:
            PSNR值
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    def calculate_ssim(self, img1, img2, window_size=11, size_average=True):
        """
        计算结构相似性指数
        
        Args:
            img1: 图像1
            img2: 图像2
            window_size: 窗口大小
            size_average: 是否平均
            
        Returns:
            SSIM值
        """
        # 简化的SSIM计算
        # 对于完整实现，建议使用scikit-image或pytorch-msssim
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        mu1 = ndimage.gaussian_filter(img1, 1.5)
        mu2 = ndimage.gaussian_filter(img2, 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ndimage.gaussian_filter(img1 ** 2, 1.5) - mu1_sq
        sigma2_sq = ndimage.gaussian_filter(img2 ** 2, 1.5) - mu2_sq
        sigma12 = ndimage.gaussian_filter(img1 * img2, 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map
    
    def evaluate_batch(self, lr_batch, hr_batch):
        """
        评估一个批次的数据
        
        Args:
            lr_batch: 低分辨率批次
            hr_batch: 高分辨率批次
            
        Returns:
            评估结果字典
        """
        with torch.no_grad():
            # 移动到设备
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            # 模型推理
            sr_batch = self.model(lr_batch)
            
            # 转换为numpy
            sr_np = sr_batch.cpu().numpy()
            hr_np = hr_batch.cpu().numpy()
            
            # 计算指标
            batch_metrics = {
                'psnr': [],
                'ssim': [],
                'rmse': [],
                'mae': [],
                #'grad_loss': []
            }
            
            # 对批次中的每个样本计算指标
            for i in range(sr_np.shape[0]):
                sample_metrics = self.evaluate_sample(sr_np[i], hr_np[i])
                for key in batch_metrics:
                    batch_metrics[key].append(sample_metrics[key])
            
            # 计算批次平均
            avg_metrics = {}
            for key in batch_metrics:
                avg_metrics[key] = np.mean(batch_metrics[key])
            
            return avg_metrics, sr_np, hr_np
    
    def evaluate_sample(self, sr_img, hr_img):
        """
        评估单个样本
        
        Args:
            sr_img: 超分辨率结果 (C, H, W)
            hr_img: 高分辨率真值 (C, H, W)
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 计算PSNR和SSIM（对每个通道，然后取平均）
        psnr_channels = []
        ssim_channels = []
        rmse_channels = []
        mae_channels = []
        
        for c in range(sr_img.shape[0]):
            sr_channel = sr_img[c]
            hr_channel = hr_img[c]
            
            # PSNR
            psnr = self.calculate_psnr(sr_channel, hr_channel)
            psnr_channels.append(psnr)
            
            # SSIM
            ssim = self.calculate_ssim(sr_channel, hr_channel)
            ssim_channels.append(ssim)
            
            # RMSE
            rmse = np.sqrt(np.mean((sr_channel - hr_channel) ** 2))
            rmse_channels.append(rmse)
            
            # MAE
            mae = np.mean(np.abs(sr_channel - hr_channel))
            mae_channels.append(mae)
        
        metrics['psnr'] = np.mean(psnr_channels)
        metrics['ssim'] = np.mean(ssim_channels)
        metrics['rmse'] = np.mean(rmse_channels)
        metrics['mae'] = np.mean(mae_channels)
        
        # # 梯度损失（使用PyTorch）
        # sr_tensor = torch.from_numpy(sr_img).unsqueeze(0).to(self.device)
        # hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).to(self.device)
        # grad_loss = self.grad_loss_fn(sr_tensor, hr_tensor).item()
        # metrics['grad_loss'] = grad_loss
        
        return metrics
    
    def evaluate_dataset(self, dataset, batch_size=4, channels_to_visualize=None):
        """
        评估整个数据集
        
        Args:
            dataset: 测试数据集
            batch_size: 批次大小
            channels_to_visualize: 要可视化的通道索引列表
        
        Returns:
            总体评估结果
        """
        print(f"开始评估数据集，样本数: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,  # 评估时不使用多线程
            pin_memory=True
        )
        
        # 重置指标
        for key in self.metrics:
            self.metrics[key] = []
        
        # 保存一些样本用于可视化
        visualization_samples = {
            'lr': [],
            'sr': [],
            'hr': [],
            'indices': []
        }
        
        # 评估所有批次
        with torch.no_grad():
            for batch_idx, (lr_batch, hr_batch) in enumerate(tqdm(dataloader, desc="评估进度")):
                batch_metrics, sr_batch, hr_batch_np = self.evaluate_batch(lr_batch, hr_batch)
                
                # 保存指标
                for key in batch_metrics:
                    self.metrics[key].append(batch_metrics[key])
                
                # 保存前几个批次的样本用于可视化
                if batch_idx < 2:  # 保存前2个批次
                    lr_batch_np = lr_batch.cpu().numpy()
                    for i in range(min(2, sr_batch.shape[0])):  # 每个批次保存前2个样本
                        sample_idx = batch_idx * batch_size + i
                        visualization_samples['lr'].append(lr_batch_np[i])
                        visualization_samples['sr'].append(sr_batch[i])
                        visualization_samples['hr'].append(hr_batch_np[i])
                        visualization_samples['indices'].append(sample_idx)
        
        # 计算总体指标
        overall_metrics = {}
        for key in self.metrics:
            if self.metrics[key]:
                overall_metrics[key] = np.mean(self.metrics[key])
            else:
                overall_metrics[key] = 0.0
        
        print("\n" + "="*60)
        print("评估结果:")
        print("="*60)
        for key, value in overall_metrics.items():
            print(f"{key.upper()}: {value:.6f}")
        print("="*60)
        
        # 保存可视化的通道配置（如果提供了）
        if channels_to_visualize is not None:
            # 保存通道配置到文件
            config_path = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 
                                      'visualization_channels.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'channels_to_visualize': channels_to_visualize,
                    'channel_names': self.channel_names,
                    'channel_indices': [{
                        'index': i,
                        'name': self.channel_names[i] if i < len(self.channel_names) else f'Channel_{i}'
                    } for i in channels_to_visualize]
                }, f, indent=4)
            print(f"可视化通道配置已保存到: {config_path}")
        
        return overall_metrics, visualization_samples
    
    def visualize_results(self, dataset, visualization_samples, save_dir='./evaluation_results', 
                     channels_to_visualize=None, channel_names=None):
        """
        可视化评估结果
        
        Args:
            dataset: 测试数据集
            visualization_samples: 可视化样本
            save_dir: 保存目录
            channels_to_visualize: 要可视化的通道索引列表，如果为None则使用前4个通道
            channel_names: 通道名称列表，如果为None则使用默认通道名称
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 确定要显示的通道
        if channels_to_visualize is None:
            # 默认显示前4个通道
            n_channels = min(4, visualization_samples['lr'][0].shape[0])
            display_channels = list(range(n_channels))
        else:
            display_channels = channels_to_visualize
        
        # 确定通道名称
        if channel_names is None:
            channel_names = self.channel_names
        
        # 为每个样本创建可视化
        for sample_idx, (lr, sr, hr, idx) in enumerate(zip(
            visualization_samples['lr'],
            visualization_samples['sr'],
            visualization_samples['hr'],
            visualization_samples['indices'])):
                for ch_idx in display_channels:
                    # 获取通道名称
                    if ch_idx < len(channel_names):
                        ch_name = channel_names[ch_idx]
                    else:
                        ch_name = f'Channel_{ch_idx}'
                    self.visualize_sample(
                        lr, sr, hr, idx,
                        dataset,
                        channel_idx=ch_idx,
                        channel_name=ch_name,
                        save_dir=save_dir,
                        sample_idx=sample_idx
                    )
        # 生成该样本的误差分布图（所有通道合并一张）
        # 反归一化
        sr_denorm = dataset.denormalize(sr[np.newaxis, ...], is_lr=False)[0]
        hr_denorm = dataset.denormalize(hr[np.newaxis, ...], is_lr=False)[0]
        self.plot_error_distribution(sr_denorm, hr_denorm, idx, display_channels,
                                     channel_names, save_dir)
    
    def visualize_sample(self, lr, sr, hr, idx, dataset, channel_idx, channel_name,
                         save_dir='./evaluation_results', sample_idx=0):
        """
        可视化单个样本的单个通道
        """
        # 反归一化
        lr_denorm = dataset.denormalize(lr[np.newaxis, ...], is_lr=True)[0]
        sr_denorm = dataset.denormalize(sr[np.newaxis, ...], is_lr=False)[0]
        hr_denorm = dataset.denormalize(hr[np.newaxis, ...], is_lr=False)[0]
    
        # 检查通道索引是否有效
        if channel_idx >= lr_denorm.shape[0]:
            print(f"警告: 通道索引 {channel_idx} 超出数据范围，跳过")
            return
    
        # 获取通道数据
        lr_ch = lr_denorm[channel_idx]
        sr_ch = sr_denorm[channel_idx]
        hr_ch = hr_denorm[channel_idx]
    
        # LR图像保持原尺寸（不进行上采样，以实际分辨率显示）
        lr_upscaled = lr_ch
    
        # 计算该通道的PSNR和RMSE
        psnr = self.calculate_psnr(sr_ch, hr_ch)
        rmse = np.sqrt(np.mean((sr_ch - hr_ch) ** 2))
    
        # 选择颜色映射
        if channel_name.lower().startswith('t') or '温度' in channel_name.lower():
            cmap = 'hot'
        elif channel_name.lower() in ['h2', 'o2', 'h2o', 'p'] or '组分' in channel_name.lower():
            cmap = 'viridis'
        elif channel_name.lower() in ['u', 'v', 'w'] or '速度' in channel_name.lower():
            cmap = 'coolwarm'
        elif channel_name.lower() in ['rho', '压力', '密度']:
            cmap = 'plasma'
        else:
            cmap = 'hot'
    
        # 确定全局范围
        vmin = min(lr_upscaled.min(), sr_ch.min(), hr_ch.min())
        vmax = max(lr_upscaled.max(), sr_ch.max(), hr_ch.max())
    
        # 创建单个通道的四子图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
        # LR
        im0 = axes[0].imshow(lr_upscaled, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{channel_name} - LR (8x下采样)', fontsize=10)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
        # SR
        im1 = axes[1].imshow(sr_ch, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{channel_name} - SR', fontsize=10)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
        # HR
        im2 = axes[2].imshow(hr_ch, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title(f'{channel_name} - HR (真值)', fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
        # 残差
        residual = sr_ch - hr_ch
        residual_max = np.abs(residual).max()
        if residual_max > 0:
            im3 = axes[3].imshow(residual, cmap='RdBu_r',
                                  vmin=-residual_max, vmax=residual_max)
        else:
            im3 = axes[3].imshow(residual, cmap='RdBu_r')
        axes[3].set_title(f'残差\nPSNR: {psnr:.2f} dB, RMSE: {rmse:.4f}', fontsize=10)
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
        plt.suptitle(f'样本 {idx} - {channel_name}', fontsize=14)
        plt.tight_layout()
    
        # 保存图像
        save_path = os.path.join(save_dir, f'sample_{idx}_channel_{channel_idx}_{channel_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存可视化结果到: {save_path}")
        plt.close()
        
        
    def plot_error_distribution(self, sr, hr, idx, display_channels, channel_names=None, 
                               save_dir='./evaluation_results'):
        """
        绘制误差分布图
        
        Args:
            sr: 超分辨率结果
            hr: 高分辨率真值
            idx: 样本索引
            display_channels: 要显示的通道索引列表
            channel_names: 通道名称列表
            save_dir: 保存目录
        """
        n_channels = len(display_channels)
        if n_channels == 0:
            return
        
        fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
        
        if n_channels == 1:
            axes = [axes]
        
        # 如果没有提供通道名称，使用默认的
        if channel_names is None:
            channel_names = self.channel_names
        
        for i, ch_idx in enumerate(display_channels):
            # 检查通道索引是否有效
            if ch_idx >= len(channel_names):
                ch_name = f'Channel_{ch_idx}'
            else:
                ch_name = channel_names[ch_idx]
            
            # 检查通道索引是否在数据范围内
            if ch_idx >= sr.shape[0]:
                continue
                
            # 计算绝对误差
            abs_error = np.abs(sr[ch_idx] - hr[ch_idx]).flatten()
            
            # 绘制误差分布直方图
            axes[i].hist(abs_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_xlabel('绝对误差')
            axes[i].set_ylabel('频数')
            axes[i].set_title(f'{ch_name} 误差分布\n均值: {abs_error.mean():.4f}, 标准差: {abs_error.std():.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'样本 {idx} 的误差分布', fontsize=14, y=1.05)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'sample_{idx}_error_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"保存误差分布图到: {save_path}")
    
    def save_evaluation_results(self, overall_metrics, save_dir='./evaluation_results'):
        """
        保存评估结果
        
        Args:
            overall_metrics: 总体评估指标
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为JSON文件
        json_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            # 转换numpy类型为Python类型
            metrics_dict = {k: float(v) for k, v in overall_metrics.items()}
            json.dump(metrics_dict, f, indent=4)
        
        # 保存为文本文件
        txt_path = os.path.join(save_dir, 'evaluation_results.txt')
        with open(txt_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("火焰超分辨率模型评估结果\n")
            f.write("="*60 + "\n\n")
            
            f.write("模型信息:\n")
            f.write(f"  模型类型: {self.model.__class__.__name__}\n")
            if hasattr(self.model, 'in_channels'):
                f.write(f"  输入通道数: {self.model.in_channels}\n")
            if hasattr(self.model, 'out_channels'):
                f.write(f"  输出通道数: {self.model.out_channels}\n")
            if hasattr(self.model, 'scale'):
                f.write(f"  超分比例: {self.model.scale}\n")
            f.write("\n")
            
            f.write("评估指标:\n")
            for key, value in overall_metrics.items():
                f.write(f"  {key.upper()}: {value:.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"评估结果已保存到:")
        print(f"  JSON文件: {json_path}")
        print(f"  文本文件: {txt_path}")

# main 函数
def main():
    """
    主函数：加载模型和测试集，进行评估和可视化
    """
    print("="*60)
    print("火焰超分辨率模型评估")
    print("="*60)
    
    # =========================
    # 配置参数
    # =========================
    config = {
        # 模型路径
        'model_path': './checkpoints/best_model.pt',
        
        # 测试数据路径
        'test_npz_path': './test/flame_sr_test/flame_test.npz',
        
        # 归一化统计信息路径（训练集使用的）
        'normalization_stats_path': None,  # 现在从模型检查点获取
        
        # 超分比例（必须与训练时一致）
        'scale': 8,
        
        # 评估参数
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 输出目录
        'output_dir': './evaluation_results',
        
        # 可视化参数
        'num_samples_to_visualize': 3,  # 可视化的样本数量
        
        # 可视化配置
        'channels_to_visualize': [0, 1, 7, 8],  # 默认可视化温度、H2、压力、密度
        'channel_names': ['Temperature', 'H2', 'O2', 'H2O', 'u', 'v', 'w', 'p', 'rho'],
        
        # 归一化配置（从模型检查点获取）
        'norm_method': 'minmax',  # 默认值，会从检查点覆盖
    }
    
    print(f"使用设备: {config['device']}")
    print(f"超分比例: {config['scale']}")
    print(f"要可视化的通道: {config['channels_to_visualize']}")
    
    # =========================
    # 1. 加载模型
    # =========================
    print("\n[1/4] 加载模型...")
    
    if not os.path.exists(config['model_path']):
        print(f"错误: 模型文件不存在: {config['model_path']}")
        print("请确保模型路径正确，或先训练模型。")
        return
    
    # 加载检查点
    checkpoint = torch.load(config['model_path'], map_location=config['device'])
    
    # 获取模型配置（从检查点或使用默认）
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        print(f"从检查点加载模型配置")
        
        # 获取归一化方法
        if 'norm_method' in model_config:
            config['norm_method'] = model_config['norm_method']
            print(f"归一化方法: {config['norm_method']}")
        
        # 获取训练集统计信息（如果保存了）
        if 'train_stats' in checkpoint:
            train_stats = checkpoint['train_stats']
            print("从检查点加载训练集统计信息")
        else:
            # 尝试从配置中获取统计信息文件路径
            train_stats = None
            print("警告: 检查点中没有训练集统计信息")
    else:
        # 使用默认配置
        model_config = {
            'scale': config['scale'],
            'in_channels': 9,
            'out_channels': 9,
            'n_feats': 128,
            'n_blocks': 32,
            'res_scale': 0.1,
            'use_edge': True
        }
        print(f"使用默认模型配置")
        train_stats = None
    
    # 创建模型实例
    model = FlameEDSR2D(
        scale=model_config.get('scale', config['scale']),
        in_channels=model_config.get('in_channels', 9),
        out_channels=model_config.get('out_channels', 9),
        n_feats=model_config.get('n_feats', 128),
        n_blocks=model_config.get('n_blocks', 32),
        res_scale=model_config.get('res_scale', 0.1),
        use_edge=model_config.get('use_edge', True)
    )
    
    # 加载模型权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("错误: 检查点中没有找到模型权重")
        return
    
    print(f"模型加载成功")
    print(f"  训练轮次: {checkpoint.get('epoch', '未知')}")
    print(f"  最佳验证损失: {checkpoint.get('val_loss', checkpoint.get('best_val', '未知'))}")
    print(f"  归一化方法: {config['norm_method']}")
    
    # =========================
    # 2. 加载测试集
    # =========================
    print("\n[2/4] 加载测试集...")
    
    if not os.path.exists(config['test_npz_path']):
        print(f"错误: 测试集文件不存在: {config['test_npz_path']}")
        print("请确保测试集路径正确，或先生成测试集。")
        return
    
    # 创建测试数据集，使用训练集的归一化统计信息
    test_dataset = FlameTestDataset(
        test_npz_path=config['test_npz_path'],
        normalization_stats_path=config['normalization_stats_path'],
        scale=config['scale'],
        norm_method=config['norm_method'],
        stats_dict=train_stats  # 使用训练集的统计信息
    )
    
    # =========================
    # 3. 评估模型
    # =========================
    print("\n[3/4] 评估模型...")
    
    # 创建评估器
    evaluator = FlameEvaluator(model, device=config['device'])
    
    # 评估数据集
    overall_metrics, visualization_samples = evaluator.evaluate_dataset(
        test_dataset, 
        batch_size=config['batch_size'],
        channels_to_visualize=config.get('channels_to_visualize', None)
    )
    
    # =========================
    # 4. 可视化结果
    # =========================
    print("\n[4/4] 可视化结果...")
    
    # 可视化结果
    evaluator.visualize_results(
        test_dataset, 
        visualization_samples,
        save_dir=config['output_dir'],
        channels_to_visualize=config.get('channels_to_visualize', None),
        channel_names=evaluator.channel_names
    )
    
    # 保存评估结果
    evaluator.save_evaluation_results(overall_metrics, save_dir=config['output_dir'])
    
    # =========================
    # 5. 额外分析
    # =========================
    print("\n[5/5] 生成额外分析...")
    
    # 创建各通道指标对比图
    plot_channel_metrics(test_dataset, evaluator, overall_metrics, config)
    
    print("\n" + "="*60)
    print("评估完成!")
    print(f"结果保存在: {config['output_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()