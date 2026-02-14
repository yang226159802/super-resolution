import torch
import torch.nn as nn
import torch.nn.functional as F



class FlameLoss(nn.Module):
    """
    火焰超分辨率综合损失函数
    """
    def __init__(self, grad_weight=0.5, tv_weight=0.01, cons_weight=0.1, interface_weight=0.2):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.grad_loss = self.GradientLoss()
        self.tv_loss = self.TotalVariationLoss()
        self.cons_weight = cons_weight
        
        self.grad_weight = grad_weight
        self.tv_weight = tv_weight
        self.interface_weight = interface_weight  # 界面损失权重
        
    class GradientLoss(nn.Module):
        """梯度损失"""
        def __init__(self):
            super().__init__()
            self.sobel_x = torch.tensor([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.sobel_y = torch.tensor([[-1, -2, -1],
                                         [0,  0,  0],
                                         [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        def forward(self, pred, target):
            # 注册buffer到设备
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)
            
            # 计算梯度
            gx_pred = F.conv2d(pred, self.sobel_x.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
            gy_pred = F.conv2d(pred, self.sobel_y.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
            gx_target = F.conv2d(target, self.sobel_x.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
            gy_target = F.conv2d(target, self.sobel_y.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
            
            # L1梯度损失
            loss = torch.abs(gx_pred - gx_target).mean() + \
                   torch.abs(gy_pred - gy_target).mean()
            
            return loss
    
    class TotalVariationLoss(nn.Module):
        """总变分损失"""
        def forward(self, x):
            tv_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
            tv_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
            return (tv_h + tv_v) / 2
    
    def conservation_loss(self, pred, target):
        """守恒损失（质量、元素守恒）"""
        # 假设前4个通道是温度T和三种组分
        pred_species = pred[:, 1:4, :, :]  # H2, O2, H2O
        target_species = target[:, 1:4, :, :]
        
        # 1. 质量守恒：组分和接近1
        pred_sum = pred_species.sum(dim=1, keepdim=True)
        target_sum = target_species.sum(dim=1, keepdim=True)
        mass_loss = torch.abs(pred_sum - target_sum).mean()
        
        # 2. 各组分守恒
        species_loss = torch.abs(pred_species - target_species).mean()
        
        return mass_loss + species_loss
    
    def compute_interface_mask(self, temperature_field, threshold=0.3, kernel_size=5):
        """
        计算火焰界面掩码（温度梯度大的区域）
        
        Args:
            temperature_field: 温度场 (B, 1, H, W)
            threshold: 梯度阈值，控制界面区域的范围
            kernel_size: 膨胀核大小，用于扩展界面区域
        """
        # 计算温度梯度
        grad_x = temperature_field[:, :, :, 1:] - temperature_field[:, :, :, :-1]
        grad_y = temperature_field[:, :, 1:, :] - temperature_field[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 归一化梯度
        grad_magnitude_norm = (grad_magnitude - grad_magnitude.min()) / \
                             (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
        
        # 生成二值掩码
        interface_mask = (grad_magnitude_norm > threshold).float()
        
        # 使用膨胀操作扩大界面区域
        if kernel_size > 0:
            kernel = torch.ones(1, 1, kernel_size, kernel_size).to(temperature_field.device)
            interface_mask = F.conv2d(interface_mask, kernel, padding=kernel_size//2)
            interface_mask = (interface_mask > 0).float()
        
        return interface_mask
    
    def interface_weighted_loss(self, pred, target, temperature_field):
        """
        火焰界面加权损失
        
        Args:
            pred: 预测值
            target: 真实值
            temperature_field: 温度场（用于计算界面区域）
        """
        # 计算界面掩码（假设温度通道是第一个通道）
        if temperature_field.dim() == 4 and temperature_field.shape[1] > 1:
            temp_channel = temperature_field[:, 0:1, :, :]  # 取温度通道
        else:
            temp_channel = temperature_field
            
        interface_mask = self.compute_interface_mask(temp_channel)
        
        # 计算每个位置的L1损失
        l1_per_pixel = torch.abs(pred - target)
        
        # 计算加权平均损失
        # 界面区域权重 = 基础权重1.0 + interface_weight
        # 非界面区域权重 = 1.0
        weights = 1.0 + self.interface_weight * interface_mask
        
        # 扩展权重到所有通道
        if pred.shape[1] > 1:
            weights = weights.repeat(1, pred.shape[1], 1, 1)
        
        # 计算加权损失
        weighted_l1 = (l1_per_pixel * weights).mean()
        
        return weighted_l1, interface_mask.mean().item()  # 返回损失和界面区域占比
    
    def forward(self, pred, target):
        # 基础L1损失
        l1 = self.l1_loss(pred, target).mean()
        
        # 梯度损失（重点关注温度通道）
        # grad = self.grad_loss(pred[:, 0:1, :, :], target[:, 0:1, :, :])  # 温度通道
        
        # TV损失
        # tv = self.tv_loss(pred)
        
        # 守恒损失
        # cons = self.conservation_loss(pred, target)
        
        # 界面加权L1损失
        # 提取温度通道用于界面加权（第0通道是温度）
        # temperature_field = pred[:, 0:1, :, :]
        # weighted_l1, interface_ratio = self.interface_weighted_loss(pred, target, temperature_field)
        
        # 总损失
        total_loss = l1 # + self.interface_weight * weighted_l1 + self.grad_weight * grad + self.tv_weight * tv + self.cons_weight * cons      
        # 记录各项损失
        loss_dict = {
            'l1': l1.item(),
            # 'grad': grad.item(),
            #'tv': tv.item(),
            #'cons': cons.item(),
            # 'weighted_l1': weighted_l1.item(),
            # 'interface_ratio': interface_ratio,  # 界面区域占比
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict