# ===========================================
# 数据预处理：3D火焰场 -> 2D训练数据集
# ===========================================
import numpy as np
import torch
import os
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import matplotlib


class FlameDataPreprocessor:
    """
    将3D火焰场数据处理为2D切片数据集
    """
    def __init__(self, data_dict, scale_factor=16):
        """
        Args:
            data_dict: 包含以下键的字典
                - 'x', 'y', 'z': 网格坐标数组 (nx, ny, nz)
                - 'u', 'v', 'w': 速度分量 (nx, ny, nz)
                - 'p': 压力场 (nx, ny, nz)
                - 'T': 温度场 (nx, ny, nz)
                - 'rho': 密度场 (nx, ny, nz)
                - 'H2', 'O2', 'H2O': 组分质量分数 (nx, ny, nz)
            scale_factor: 超分辨率比例因子
        """
        self.data = data_dict
        self.scale = scale_factor
        self.normalization_stats = {}
    
    def extract_complete_slices(self, axis='z', sampling_method='uniform', num_slices=None):
        """
        沿指定轴提取完整的2D切面
        
        Args:
            axis: 'x', 'y' 或 'z'，切片方向
            sampling_method: 'uniform' 等间距采样, 'random' 随机采样
            num_slices: 采样切片数量，None表示使用所有切片
            
        Returns:
            slices_dict: 包含所有变量的完整2D切面字典
        """
        print(f"沿{axis}轴提取完整2D切面，采样方式: {sampling_method}...")
        
        # 获取网格尺寸
        nx, ny, nz = self.data['T'].shape
        
        # 根据切片轴确定切片总数
        if axis == 'x':
            n_total_slices = nx
        elif axis == 'y':
            n_total_slices = ny
        else:  # 'z'
            n_total_slices = nz
        
        # 确定采样切片索引
        if num_slices is None or num_slices >= n_total_slices:
            # 使用所有切片
            slice_indices = range(n_total_slices)
        elif sampling_method == 'uniform':
            # 等间距采样
            step = n_total_slices // num_slices
            slice_indices = range(0, n_total_slices, step)[:num_slices]
        elif sampling_method == 'random':
            # 随机采样
            slice_indices = np.random.choice(n_total_slices, num_slices, replace=False)
            slice_indices.sort()
        else:
            raise ValueError(f"未知的采样方式: {sampling_method}")
        
        slices = {
            'T': [], 'H2': [], 'O2': [], 'H2O': [],
            'u': [], 'v': [], 'w': [], 'p': [], 'rho': []
        }
        
        for i in tqdm(slice_indices):
            if axis == 'x':
                # x方向切片 (y-z平面)
                slices['T'].append(self.data['T'][i, :, :])
                slices['H2'].append(self.data['H2'][i, :, :])
                slices['O2'].append(self.data['O2'][i, :, :])
                slices['H2O'].append(self.data['H2O'][i, :, :])
                slices['u'].append(self.data['u'][i, :, :])
                slices['v'].append(self.data['v'][i, :, :])
                slices['w'].append(self.data['w'][i, :, :])
                slices['p'].append(self.data['p'][i, :, :])
                slices['rho'].append(self.data['rho'][i, :, :])
                
            elif axis == 'y':
                # y方向切片 (x-z平面)
                slices['T'].append(self.data['T'][:, i, :])
                slices['H2'].append(self.data['H2'][:, i, :])
                slices['O2'].append(self.data['O2'][:, i, :])
                slices['H2O'].append(self.data['H2O'][:, i, :])
                slices['u'].append(self.data['u'][:, i, :])
                slices['v'].append(self.data['v'][:, i, :])
                slices['w'].append(self.data['w'][:, i, :])
                slices['p'].append(self.data['p'][:, i, :])
                slices['rho'].append(self.data['rho'][:, i, :])
                
            else:  # 'z'
                # z方向切片 (x-y平面)
                slices['T'].append(self.data['T'][:, :, i])
                slices['H2'].append(self.data['H2'][:, :, i])
                slices['O2'].append(self.data['O2'][:, :, i])
                slices['H2O'].append(self.data['H2O'][:, :, i])
                slices['u'].append(self.data['u'][:, :, i])
                slices['v'].append(self.data['v'][:, :, i])
                slices['w'].append(self.data['w'][:, :, i])
                slices['p'].append(self.data['p'][:, :, i])
                slices['rho'].append(self.data['rho'][:, :, i])
        
        # 转换为numpy数组
        for key in slices:
            slices[key] = np.array(slices[key])
            
        print(f"提取了 {len(slice_indices)} 个完整切面")
        return slices
    
    def create_sr_pairs_complete_slices(self, slices_dict):
        """
        创建超分辨率训练对：完整的低分辨率(LR)和高分辨率(HR)切面对
        
        Args:
            slices_dict: 归一化后的完整切面数据字典
            
        Returns:
            lr_data_dict: 低分辨率切面字典（多个变量通道）
            hr_data_dict: 高分辨率切面字典（多个变量通道）
        """
        print("创建完整切面的超分辨率训练对...")
        
        # 确定第一个变量的形状作为参考
        first_key = list(slices_dict.keys())[0]
        n_slices, h, w = slices_dict[first_key].shape
        
        # 计算LR尺寸，确保可以整除
        lr_h = h // self.scale
        lr_w = w // self.scale
        
        # 调整HR尺寸，使其等于LR尺寸 × scale
        hr_h_adjusted = lr_h * self.scale
        hr_w_adjusted = lr_w * self.scale
        
        print(f"原始尺寸: HR={h}x{w}, LR={lr_h}x{lr_w}")
        print(f"调整后: HR={hr_h_adjusted}x{hr_w_adjusted}")
        
        # 为每个变量创建LR和HR对
        lr_data_list = []
        hr_data_list = []
        
        for var_name, data in slices_dict.items():
            var_lr = []
            var_hr = []
            
            for i in range(n_slices):
                slice_2d = data[i]
                
                # 1. 调整HR尺寸到可整除的尺寸
                if h != hr_h_adjusted or w != hr_w_adjusted:
                    # 使用双线性插值调整HR尺寸
                    hr_slice = ndimage.zoom(
                        slice_2d,
                        (hr_h_adjusted / h, hr_w_adjusted / w),
                        order=1  # 双线性插值
                    )
                else:
                    hr_slice = slice_2d
                
                # 2. 保存调整后的HR切面
                var_hr.append(hr_slice)
                
                # 3. 生成对应的LR切面（直接下采样，不再上采样）
                # 使用双线性插值下采样到LR尺寸
                lr_slice = ndimage.zoom(
                    hr_slice,
                    (1/self.scale, 1/self.scale),
                    order=1
                )
                
                # 注意：这里不再上采样回原始尺寸
                # LR尺寸应该是(lr_h, lr_w)
                var_lr.append(lr_slice)
            
            lr_data_list.append(np.array(var_lr))
            hr_data_list.append(np.array(var_hr))
        
        # 合并所有变量通道
        # 维度: (n_slices, h, w) -> (n_slices, n_channels, h, w)
        lr_combined = np.stack(lr_data_list, axis=1)  # (n_slices, n_channels, lr_h, lr_w)
        hr_combined = np.stack(hr_data_list, axis=1)  # (n_slices, n_channels, hr_h, hr_w)
        
        print(f"数据形状: LR={lr_combined.shape}, HR={hr_combined.shape}")
        print(f"LR->HR比例: {lr_combined.shape[2:]} -> {hr_combined.shape[2:]}")
        
        # 验证尺寸关系
        assert hr_combined.shape[2] == lr_combined.shape[2] * self.scale, \
            f"HR高度{hr_combined.shape[2]} != LR高度{lr_combined.shape[2]} × {self.scale}"
        assert hr_combined.shape[3] == lr_combined.shape[3] * self.scale, \
            f"HR宽度{hr_combined.shape[3]} != LR宽度{lr_combined.shape[3]} × {self.scale}"
        
        return lr_combined, hr_combined
    
    def normalize_complete_slices(self, slices_dict, mode='per_channel'):
        """
        归一化完整切面数据
        
        Args:
            slices_dict: 切面数据字典
            mode: 'per_channel' 每个通道独立归一化
                  'global' 全局归一化
                  
        Returns:
            normalized_dict: 归一化后的数据字典
            stats_dict: 归一化统计信息
        """
        print("归一化完整切面数据...")
        
        normalized = {}
        stats = {}
        
        for key, data in slices_dict.items():
            # 统计信息
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            stats[key] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
            
            # 归一化
            if mode == 'per_channel':
                # Z-score归一化
                normalized[key] = (data - mean_val) / (std_val + 1e-8)
            elif mode == 'global':
                # Min-max归一化到[0,1]
                normalized[key] = (data - min_val) / (max_val - min_val + 1e-8)
            else:
                raise ValueError(f"未知的归一化模式: {mode}")
        
        self.normalization_stats = stats
        return normalized, stats
    
    def save_complete_dataset(self, lr_data, hr_data, save_dir='./flame_sr_complete'):
        """
        保存完整切面的训练数据集
        
        Args:
            lr_data: 低分辨率完整切面数据
            hr_data: 高分辨率完整切面数据
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 分割训练集和验证集 (80%训练, 20%验证)
        n_samples = lr_data.shape[0]
        indices = np.random.permutation(n_samples)
        split_idx = int(0.8 * n_samples)
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        # 保存训练集
        train_lr = lr_data[train_idx]
        train_hr = hr_data[train_idx]
        
        # 保存验证集
        val_lr = lr_data[val_idx]
        val_hr = hr_data[val_idx]
        
        # 保存为npz文件
        np.savez(
            os.path.join(save_dir, 'flame_complete_train.npz'),
            lr=train_lr,
            hr=train_hr
        )
        
        np.savez(
            os.path.join(save_dir, 'flame_complete_val.npz'),
            lr=val_lr,
            hr=val_hr
        )
        
        # 保存归一化统计信息
        with open(os.path.join(save_dir, 'normalization_stats.pkl'), 'wb') as f:
            pickle.dump(self.normalization_stats, f)
        
        print(f"完整切面数据集已保存到 {save_dir}")
        print(f"训练集: {train_lr.shape[0]} 个样本")
        print(f"验证集: {val_lr.shape[0]} 个样本")
        print(f"数据形状: LR={train_lr.shape}, HR={train_hr.shape}")
        
        return train_lr.shape[0], val_lr.shape[0]
    
def main():
    """
    主函数：处理原始火焰数据并训练超分辨率模型
    """
    print("=" * 60)
    print("火焰超分辨率训练程序")
    print("=" * 60)
    
    #加载数据
    x_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/grid/X_m.dat'
    y_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/grid/Y_m.dat'
    z_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/grid/Z_m.dat'
    #variable_data
    T_path="G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/T_K_id000.dat"
    RHO_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/RHO_kgm-3_id000.dat'
    Y_H2_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/YH2_id000.dat'
    Y_O2_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/YO2_id000.dat'
    Y_H2O_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/YH2O_id000.dat'
    p_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/P_Pa_id000.dat'
    u_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/UX_ms-1_id000.dat'
    v_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/UY_ms-1_id000.dat'
    w_path='G:/毕设数据集/Premixed Flames H2 Air DNS/Premixed Flames H2 Air DNS (phi = 0.35)/data/UZ_ms-1_id000.dat'
    
    
    x = np.fromfile(x_path,dtype='<f4')
    y = np.fromfile(y_path,dtype='<f4')
    z = np.fromfile(z_path,dtype='<f4')
    T = np.fromfile(T_path,dtype='<f4')
    RHO = np.fromfile(RHO_path,dtype='<f4')
    Y_H2 = np.fromfile(Y_H2_path,dtype='<f4')
    Y_O2 = np.fromfile(Y_O2_path,dtype='<f4')
    Y_H2O = np.fromfile(Y_H2O_path,dtype='<f4')
    p = np.fromfile(p_path,dtype='<f4')
    u = np.fromfile(u_path,dtype='<f4')
    v = np.fromfile(v_path,dtype='<f4')
    w = np.fromfile(w_path,dtype='<f4')
    
    T = T.reshape(len(x),len(y),len(z))
    RHO = RHO.reshape(len(x),len(y),len(z))
    Y_H2 = Y_H2.reshape(len(x),len(y),len(z))
    Y_O2 = Y_O2.reshape(len(x),len(y),len(z))
    Y_H2O = Y_H2O.reshape(len(x),len(y),len(z))
    p = p.reshape(len(x),len(y),len(z))
    u = u.reshape(len(x),len(y),len(z))
    v = v.reshape(len(x),len(y),len(z))
    w = w.reshape(len(x),len(y),len(z))
    
    # 示例数据加载（需要用户根据实际情况修改）
    data_dict = {
        'x': x,  # (nx, ny, nz) 网格坐标
        'y': y,
        'z': z,
        'u': u,  # (nx, ny, nz) 速度分量
        'v': v,
        'w': w,
        'p': p,  # 压力
        'T': T,  # 温度
        'rho': RHO,  # 密度
        'H2': Y_H2,   # H2质量分数
        'O2': Y_O2,   # O2质量分数
        'H2O': Y_H2O  # H2O质量分数
    }
    
    
    # 提示用户输入数据
    print("\n请确保已将原始火焰数据加载到data_dict字典中")
    print("包含以下键: x, y, z, u, v, w, p, T, rho, H2, O2, H2O")
    print("每个字段都是三维numpy数组")
    
    # 这里应该从用户获取数据，但根据要求"不用交互"
    # 所以假设数据已经准备好，直接调用预处理
    
    # 1. 数据预处理
    print("数据预处理...")
    preprocessor = FlameDataPreprocessor(data_dict, scale_factor=8)
    
    # 提取完整2D切面
    slices = preprocessor.extract_complete_slices(
        axis='z', 
        sampling_method='uniform', 
        num_slices=200  # 可以根据需要调整
    )
    
    # # 归一化
    # normalized_slices, stats = preprocessor.normalize_complete_slices(
    #     slices, 
    #     mode='per_channel'
    # )
    
    # 创建SR训练对（完整切面）
    lr_data, hr_data = preprocessor.create_sr_pairs_complete_slices(slices)
    
    # 保存数据集
    n_train, n_val = preprocessor.save_complete_dataset(
        lr_data, 
        hr_data, 
        save_dir='./flame_sr_complete'
    )
    
if __name__ == "__main__":
    main()