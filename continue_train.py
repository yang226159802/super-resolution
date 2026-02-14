# continue_train.py
# -*- coding: utf-8 -*-
"""
统一火焰超分辨率训练脚本
支持初次训练和从检查点继续训练
"""

import os
import csv
import random
import math
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

# 自定义模块
from flame_dataset1 import FlameDataset
from flame_edsr2d import FlameEDSR2D
from flame_loss1 import FlameLoss

# 全局设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='火焰超分辨率统一训练脚本')
    
    # -------------------- 基础配置 --------------------
    parser.add_argument('--train_npz', type=str, default='./flame_sr_complete/flame_complete_train.npz',
                        help='训练集npz路径')
    parser.add_argument('--val_npz', type=str, default='./flame_sr_complete/flame_complete_val.npz',
                        help='验证集npz路径')
    parser.add_argument('--scale', type=int, default=8,
                        help='上采样倍数 (默认: 8)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小 (默认: 1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='总训练轮数 (初次训练时有效, 继续训练时会被检查点覆盖)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率 (默认: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--norm_method', type=str, default='minmax',
                        choices=['zscore', 'minmax', 'none'],
                        help='归一化方法 (默认: minmax)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='随机种子 (默认: 2025)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数 (默认: 0)')
    
    # -------------------- 模型结构 --------------------
    parser.add_argument('--in_channels', type=int, default=9,
                        help='输入通道数 (默认: 9)')
    parser.add_argument('--out_channels', type=int, default=9,
                        help='输出通道数 (默认: 9)')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='特征通道数 (默认: 128)')
    parser.add_argument('--n_blocks', type=int, default=16,
                        help='残差块数量 (默认: 16)')
    parser.add_argument('--use_edge', action='store_true', default=True,
                        help='是否使用边缘引导 (默认: True)')
    
    # -------------------- 损失权重 --------------------
    parser.add_argument('--grad_weight', type=float, default=0.0,
                        help='梯度损失权重 (默认: 0.0)')
    parser.add_argument('--tv_weight', type=float, default=0.0,
                        help='总变分损失权重 (默认: 0.0)')
    parser.add_argument('--cons_weight', type=float, default=0.0,
                        help='守恒损失权重 (默认: 0.0)')
    parser.add_argument('--interface_weight', type=float, default=0.2,
                        help='界面加权损失权重 (默认: 0.2)')
    
    # -------------------- 路径配置 --------------------
    parser.add_argument('--log_dir', type=str, default='./runs/flame_sr_complete',
                        help='TensorBoard日志目录 (当前未使用)')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='可视化结果保存目录')
    parser.add_argument('--csv_log', type=str, default='./logs/flame_sr_complete_log.csv',
                        help='CSV训练日志路径')
    
    # -------------------- 可视化与保存 --------------------
    parser.add_argument('--vis_freq', type=int, default=10,
                        help='可视化频率（每N个epoch）')
    
    # -------------------- 继续训练相关 --------------------
    parser.add_argument('--resume', action='store_true',default=True,
                        help='是否从检查点继续训练')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/final_model.pt',
                        help='继续训练的检查点路径（当resume=True时必须提供）')
    parser.add_argument('--resume_epochs', type=int, default=50,
                        help='继续训练时额外训练的epoch数（默认: 50）')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_experiment(args):
    """
    根据参数设置实验环境，生成配置字典，创建目录
    返回: config, device
    """
    # 从命令行参数构建配置字典
    config = vars(args).copy()
    
    # 如果是继续训练，总epochs会在加载检查点时重新计算，此处暂时保留
    if args.resume:
        config['epochs'] = args.resume_epochs  # 临时存放额外训练epoch数
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建必要目录
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['csv_log']), exist_ok=True)
    os.makedirs(config['vis_dir'], exist_ok=True)
    
    # 打印关键配置
    print("\n========== 实验配置 ==========")
    for key in ['scale', 'batch_size', 'lr', 'norm_method', 'n_feats', 'n_blocks', 'use_edge']:
        print(f"{key}: {config[key]}")
    print("===============================\n")
    
    return config, device


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载检查点，返回训练状态字典
    """
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载配置（用于覆盖）
    saved_config = checkpoint.get('config', {})
    
    # 加载训练历史
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    
    # 加载训练状态
    start_epoch = checkpoint['epoch'] + 1  # 下一个epoch开始
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # 加载优化器和调度器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'saved_config': saved_config,
        'start_epoch': start_epoch,
        'best_val_loss': best_val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'checkpoint_path': checkpoint_path
    }


def init_or_resume(args, config, device):
    """
    根据args.resume决定是初始化新模型还是从检查点恢复
    返回: (model, optimizer, scheduler, scaler, training_state)
    """
    # 模型初始化（使用配置中的结构参数）
    model = FlameEDSR2D(
        scale=config['scale'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_feats=config['n_feats'],
        n_blocks=config['n_blocks'],
        use_edge=config['use_edge']
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # 损失函数
    criterion = FlameLoss(
        grad_weight=config['grad_weight'],
        tv_weight=config['tv_weight'],
        cons_weight=config['cons_weight'],
        interface_weight=config['interface_weight']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],  # 稍后会根据实际情况调整
        eta_min=1e-6
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    training_state = {
        'start_epoch': 1,
        'best_val_loss': float('inf'),
        'train_loss_history': [],
        'val_loss_history': [],
        'criterion': criterion,
        'checkpoint_path': None
    }
    
    # 如果继续训练，加载检查点
    if args.resume:
        if args.checkpoint is None:
            raise ValueError("启用 --resume 时必须指定 --checkpoint 路径")
        
        # 加载检查点状态
        ckpt_state = load_checkpoint(args.checkpoint, model, optimizer, scheduler)
        
        # 更新训练状态
        training_state.update(ckpt_state)
        
        # 重要：使用检查点中保存的配置覆盖当前配置（除非用户在命令行显式覆盖）
        # 但命令行参数优先级更高（已在config中）
        # 我们将需要覆盖的配置项打印出来
        if 'saved_config' in ckpt_state:
            saved_cfg = ckpt_state['saved_config']
            print("从检查点加载的原始配置:")
            for k in ['scale', 'norm_method', 'in_channels', 'out_channels', 'n_feats', 'n_blocks', 'use_edge']:
                if k in saved_cfg:
                    print(f"  {k}: {saved_cfg[k]} (当前: {config.get(k, '未知')})")
                    # 如果用户没有在命令行显式指定，可以自动使用保存的配置
                    # 简单起见，这里强制使用保存的配置，以保证一致性
                    config[k] = saved_cfg[k]
        
        # 重新计算总epoch数
        # 原始总epoch = 检查点中的epoch + 额外epoch数
        total_epochs = ckpt_state['start_epoch'] - 1 + args.resume_epochs
        config['epochs'] = total_epochs
        
        # 重新设置调度器的T_max（从当前epoch开始到总epoch数）
        scheduler.T_max = args.resume_epochs
        scheduler.last_epoch = ckpt_state['start_epoch'] - 2  # 内部从0开始计数
        scheduler.step()  # 恢复到正确位置
        
        print(f"继续训练: 从epoch {training_state['start_epoch']} 到 {total_epochs}")
        print(f"最佳验证损失: {training_state['best_val_loss']:.6f}")
    
    else:
        # 初次训练
        print("初次训练，初始化新模型")
        training_state['start_epoch'] = 1
        training_state['best_val_loss'] = float('inf')
    
    return model, optimizer, scheduler, scaler, training_state, config


def load_datasets(config, training_state):
    """
    加载训练集和验证集
    如果继续训练，验证集使用训练集的统计信息（从训练集重新计算或从检查点？）
    这里统一重新计算训练集的统计信息，保证一致性
    """
    print("\n加载数据集...")
    train_dataset = FlameDataset(
        config['train_npz'],
        normalize=True,
        norm_method=config['norm_method']
    )
    train_stats = train_dataset.get_stats_dict()
    
    val_dataset = FlameDataset(
        config['val_npz'],
        normalize=True,
        norm_method=config['norm_method'],
        stats_dict=train_stats
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    return train_loader, val_loader, train_dataset, val_dataset


def save_visualization(model, fixed_lr, fixed_hr, epoch, val_dataset, vis_dir):
    """保存温度场可视化"""
    model.eval()
    with torch.no_grad():
        fixed_sr = model(fixed_lr)
    
    # 反归一化
    lr_np = val_dataset.denormalize(fixed_lr.cpu().numpy(), is_lr=True)[0]
    hr_np = val_dataset.denormalize(fixed_hr.cpu().numpy(), is_lr=False)[0]
    sr_np = val_dataset.denormalize(fixed_sr.cpu().numpy(), is_lr=False)[0]
    
    # 取温度通道（第0通道）
    lr_temp = lr_np[0]
    hr_temp = hr_np[0]
    sr_temp = sr_np[0]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    vmin = min(lr_temp.min(), sr_temp.min(), hr_temp.min())
    vmax = max(lr_temp.max(), sr_temp.max(), hr_temp.max())
    
    im1 = axes[0, 0].imshow(lr_temp, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('低分辨率输入 (LR)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(hr_temp, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('高分辨率真实值 (HR)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 0].imshow(sr_temp, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('超分辨率重建 (SR)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    error = sr_temp - hr_temp
    error_max = max(abs(error.min()), abs(error.max()))
    im4 = axes[1, 1].imshow(error, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    axes[1, 1].set_title('重建误差 (SR - HR)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    #title_prefix = '继续训练' if args.resume else '训练'
    plt.suptitle(f'火焰温度场对比 (Epoch {epoch})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(vis_dir, f'field_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存: {save_path}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, desc_prefix):
    """单个epoch的训练过程"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    pbar = tqdm(loader, desc=desc_prefix)
    for lr_imgs, hr_imgs in pbar:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(lr_imgs)
            loss, loss_dict = criterion(preds, hr_imgs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = lr_imgs.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{loss_dict['l1']:.4f}"
        })
    
    return total_loss / max(1, n_samples)


def validate(model, loader, criterion, device):
    """验证过程"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="验证")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            preds = model(lr_imgs)
            loss, _ = criterion(preds, hr_imgs)
            
            batch_size = lr_imgs.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / max(1, n_samples)


def plot_loss_curves(train_losses, val_losses, vis_dir, start_epoch=None):
    """绘制损失曲线，并标记继续训练起点（如果有）"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 普通坐标
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    if start_epoch and start_epoch > 1:
        ax1.axvline(x=start_epoch-1, color='g', linestyle='--', alpha=0.7, label=f'继续训练起点')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 对数坐标
    ax2.semilogy(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    ax2.semilogy(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    if start_epoch and start_epoch > 1:
        ax2.axvline(x=start_epoch-1, color='g', linestyle='--', alpha=0.7, label=f'继续训练起点')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('对数尺度损失曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(vis_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, config,
                    train_loss_history, val_loss_history, is_best=False,
                    original_checkpoint=None, save_dir='./checkpoints'):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': best_val_loss,
        'config': config,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }
    if original_checkpoint:
        checkpoint['original_checkpoint'] = original_checkpoint
    
    # 文件名
    if is_best:
        fname = 'best_model.pt'
    else:
        fname = f'checkpoint_epoch_{epoch:04d}.pt'
    
    path = os.path.join(save_dir, fname)
    torch.save(checkpoint, path)
    print(f"检查点已保存: {path}")
    return path


def main():
    args = parse_args()
    
    # 1. 设置随机种子
    set_seed(args.seed)
    
    # 2. 实验环境配置
    config, device = setup_experiment(args)
    
    # 3. 模型、优化器、调度器初始化或恢复
    model, optimizer, scheduler, scaler, training_state, config = init_or_resume(args, config, device)
    criterion = training_state['criterion']
    
    # 4. 加载数据集
    train_loader, val_loader, train_dataset, val_dataset = load_datasets(config, training_state)
    
    # 5. 获取固定样本用于可视化（从验证集取第一个样本）
    fixed_lr, fixed_hr = None, None
    for lr, hr in val_loader:
        fixed_lr, fixed_hr = lr[0:1].to(device), hr[0:1].to(device)
        break
    
    # 6. CSV日志设置
    if args.resume:
        # 继续训练时，创建新的日志文件（保留原日志不变）
        base, ext = os.path.splitext(config['csv_log'])
        log_path = f"{base}_continued{ext}"
    else:
        log_path = config['csv_log']
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    
    # 7. 训练循环
    start_epoch = training_state['start_epoch']
    total_epochs = config['epochs']
    best_val_loss = training_state['best_val_loss']
    train_loss_history = training_state['train_loss_history']
    val_loss_history = training_state['val_loss_history']
    
    print(f"\n开始训练，起始epoch: {start_epoch}, 总epoch数: {total_epochs}")
    for epoch in range(start_epoch, total_epochs + 1):
        # 训练一个epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, f"Epoch {epoch}/{total_epochs} [训练]"
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # 写入CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.6e}"])
        
        # 打印进度
        print(f"Epoch {epoch}/{total_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.2e}")
        
        # 定期可视化
        if epoch % config['vis_freq'] == 0 and fixed_lr is not None:
            save_visualization(model, fixed_lr, fixed_hr, epoch, val_dataset, config['vis_dir'])
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                epoch, model, optimizer, scheduler, best_val_loss, config,
                train_loss_history, val_loss_history, is_best=True,
                original_checkpoint=training_state.get('checkpoint_path'),
                save_dir=config['model_save_dir']
            )
        
        # 定期保存检查点（每10个epoch）
        if epoch % 10 == 0:
            save_checkpoint(
                epoch, model, optimizer, scheduler, best_val_loss, config,
                train_loss_history, val_loss_history, is_best=False,
                original_checkpoint=training_state.get('checkpoint_path'),
                save_dir=config['model_save_dir']
            )
    
    # 8. 训练结束，保存最终模型
    final_path = os.path.join(config['model_save_dir'], 'final_model.pt')
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'final_val_loss': val_loss_history[-1] if val_loss_history else None
    }, final_path)
    print(f"最终模型已保存: {final_path}")
    
    # 9. 绘制损失曲线
    plot_loss_curves(train_loss_history, val_loss_history, config['vis_dir'],
                     start_epoch=start_epoch if args.resume else None)
    
    # 10. 打印总结
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"总训练epoch数: {len(train_loss_history)}")
    best_epoch = val_loss_history.index(min(val_loss_history)) + 1   # 获取最佳epoch
    print(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"最终验证损失: {val_loss_history[-1]:.6f}")
    print(f"模型保存目录: {config['model_save_dir']}")
    print(f"可视化目录: {config['vis_dir']}")
    print(f"训练日志: {log_path}")
    print("="*60)
    
    return model, train_loss_history, val_loss_history


if __name__ == "__main__":
    # 启动训练
    model, train_losses, val_losses = main()