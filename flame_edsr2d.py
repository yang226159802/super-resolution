import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlameMSResBlock(nn.Module):
    """
    火焰专用的多尺度残差块
    """
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_d = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        y1 = self.act(self.conv1(x))
        y2 = self.act(self.conv_d(x))
        y = self.fuse(torch.cat([y1, y2], dim=1))
        y = self.conv2(y)
        return x + self.res_scale * y

class FlameEdgeGuide(nn.Module):
    """
    火焰边缘引导：基于温度梯度提取火焰锋面特征
    """
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        # 针对温度通道特别处理
        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1)
        )
        
        # 其他通道的特征提取
        self.other_conv = nn.Sequential(
            nn.Conv2d(in_ch-1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1)
        )
        
        # 融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(32, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        # 假设温度是第一个通道
        temp = x[:, 0:1, :, :]
        other = x[:, 1:, :, :]
        
        # 计算温度梯度
        gx = temp[:, :, :, 1:] - temp[:, :, :, :-1]
        gy = temp[:, :, 1:, :] - temp[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        g_temp = torch.sqrt(gx * gx + gy * gy + 1e-8)
        
        # 提取特征
        feat_temp = self.temp_conv(g_temp)
        feat_other = self.other_conv(other)
        
        # 融合特征
        combined = torch.cat([feat_temp, feat_other], dim=1)
        return self.fuse(combined)

class FlameUpsampler(nn.Sequential):
    """
    火焰上采样模块
    """
    def __init__(self, scale, n_feats):
        m = []
        assert math.log2(scale).is_integer(), "scale 必须是 2 的幂（如 2,4,8,16）"
        for _ in range(int(math.log2(scale))):
            m += [
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        super().__init__(*m)

class FlamePhysicalConstraint(nn.Module):
    """
    火焰物理约束层
    """
    def forward(self, x, normalization_stats=None):
        """
        Args:
            x: 模型输出，假设通道顺序为 [T, H2, O2, H2O, u, v, w, p, rho]
            normalization_stats: 归一化统计信息
        """
        if normalization_stats is None:
            return x
        
        # 反归一化
        denorm_x = self.denormalize(x, normalization_stats)
        
        # 应用物理约束
        # 1. 温度非负（假设K为单位）
        denorm_x[:, 0] = torch.clamp(denorm_x[:, 0], min=300.0)  # 最低300K
        
        # 2. 组分质量分数在[0,1]之间
        denorm_x[:, 1:4] = torch.clamp(denorm_x[:, 1:4], min=0.0, max=1.0)
        
        # 3. 确保组分守恒（H2 + O2 + H2O ≈ 1）
        species_sum = denorm_x[:, 1:4].sum(dim=1, keepdim=True)
        denorm_x[:, 1:4] = denorm_x[:, 1:4] / (species_sum + 1e-8)
        
        # 4. 密度非负
        denorm_x[:, -1] = torch.clamp(denorm_x[:, -1], min=0.0)
        
        # 重新归一化
        return self.renormalize(denorm_x, normalization_stats)
    
    def denormalize(self, x, stats):
        """反归一化"""
        denorm = torch.zeros_like(x)
        for i, (key, stat) in enumerate(stats.items()):
            if i >= x.shape[1]:  # 防止索引越界
                break
            # 假设使用Z-score归一化
            denorm[:, i] = x[:, i] * stat['std'] + stat['mean']
        return denorm
    
    def renormalize(self, x, stats):
        """重新归一化"""
        norm = torch.zeros_like(x)
        for i, (key, stat) in enumerate(stats.items()):
            if i >= x.shape[1]:  # 防止索引越界
                break
            norm[:, i] = (x[:, i] - stat['mean']) / (stat['std'] + 1e-8)
        return norm

class FlameEDSR2D(nn.Module):
    """
    火焰专用的增强版EDSR2D模型
    """
    def __init__(self, scale=16, in_channels=9, out_channels=9,
                 n_feats=128, n_blocks=32, res_scale=0.1, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        self.in_channels = in_channels
        self.scale = scale  # 保存scale参数
        
        # 输入头
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        
        # 残差块主体
        blocks = [FlameMSResBlock(n_feats, res_scale=res_scale) for _ in range(n_blocks)]
        self.body = nn.Sequential(*blocks, nn.Conv2d(n_feats, n_feats, 3, 1, 1))
        
        # 边缘引导
        if self.use_edge:
            edge_ch = max(32, n_feats // 4)
            self.edge = FlameEdgeGuide(in_ch=in_channels, out_ch=edge_ch)
            self.fuse = nn.Conv2d(n_feats + edge_ch, n_feats, 1, 1, 0)
        else:
            self.edge = None
            self.fuse = None
        
        # 上采样
        self.ups = FlameUpsampler(scale, n_feats)
        
        # 输出尾
        self.tail = nn.Conv2d(n_feats, out_channels, 3, 1, 1)
        
        # 物理约束层
        self.physical_constraint = FlamePhysicalConstraint()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, normalization_stats=None):
        # 记录输入尺寸
        input_shape = x.shape
        # print(f"输入尺寸: {input_shape}")  # 调试用
        
        # 提取特征
        h = self.head(x)
        b = self.body(h)
        
        # 边缘引导融合
        if self.use_edge:
            e = self.edge(x)
            h = self.fuse(torch.cat([h + b, e], dim=1))
        else:
            h = h + b
        
        # 上采样
        y = self.ups(h)
        y = self.tail(y)
        
        # 验证输出尺寸
        output_shape = y.shape
        expected_h = input_shape[2] * self.scale
        expected_w = input_shape[3] * self.scale
        # print(f"输出尺寸: {output_shape}, 期望尺寸: (..., {expected_h}, {expected_w})")  # 调试用
        
        # 应用物理约束
        if normalization_stats is not None:
            y = self.physical_constraint(y, normalization_stats)
        
        return y