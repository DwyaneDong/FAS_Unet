import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_fas_unet_dataset_h5py(mat_filename):
    with h5py.File(mat_filename, "r") as f:
        print("HDF5文件中的变量:")
        
        # 修改读取基本参数的方式
        try:
            # 方法1：直接读取为numpy数组
            M_fixed = np.array(f["M_fixed"]).item()
            K_fixed = np.array(f["K_fixed"]).item()
        except Exception as e:
            print(f"使用备选方法读取参数: {str(e)}")
            # 方法2：尝试其他读取方式
            M_fixed = int(np.array(f["M_fixed"]).squeeze())
            K_fixed = int(np.array(f["K_fixed"]).squeeze())

        # 读取通道数据并指定数据类型
        masked_channels = np.array(f["all_masked_tensor"], dtype=np.float64)
        full_channels = np.array(f["all_full_tensor"], dtype=np.float64)
        #masked_channels = np.array(f["all_masked_array"], dtype=np.float64)
        #full_channels = np.array(f["all_full_array"], dtype=np.float64)
        
        # 转换维度顺序
        masked_channels = np.transpose(masked_channels, (3, 0, 1, 2))
        full_channels = np.transpose(full_channels, (3, 0, 1, 2))
        
        print(f"M_fixed: {M_fixed}, K_fixed: {K_fixed}")
        print(f"最终数组形状: {masked_channels.shape}")
        print(f"完整信道形状: {full_channels.shape}")
        
        return masked_channels, full_channels, M_fixed, K_fixed


class FASUNetDataset(Dataset):
    """流体天线信道U-Net数据集类"""

    def __init__(self, mat_filename, transform=None, normalize=True):
        self.transform = transform
        self.normalize = normalize

        # 加载数据
        result = load_fas_unet_dataset_h5py(mat_filename)
        if result is None:
            raise ValueError("数据加载失败")

        self.masked_channels, self.full_channels, self.M, self.K = result

        print(f"数据集统计:")
        print(f"  样本数量: {len(self.masked_channels)}")
        print(f"  网格尺寸: {self.M} x {self.K}")
        print(f"  单个样本形状: {self.masked_channels[0].shape}")

        # 计算统计信息用于归一化
        if self.normalize:
            self._compute_stats()

    def _compute_stats(self):
        """计算数据统计信息"""
        # 使用完整信道数据计算统计信息
        real_data = self.full_channels[:, :, :, 0]
        imag_data = self.full_channels[:, :, :, 1]

        self.real_mean = np.mean(real_data)
        self.real_std = np.std(real_data)
        self.imag_mean = np.mean(imag_data)
        self.imag_std = np.std(imag_data)

        print(f"数据统计:")
        print(f"  实部 - 均值: {self.real_mean:.6f}, 标准差: {self.real_std:.6f}")
        print(f"  虚部 - 均值: {self.imag_mean:.6f}, 标准差: {self.imag_std:.6f}")

        # 避免除零
        if self.real_std < 1e-10:
            self.real_std = 1.0
        if self.imag_std < 1e-10:
            self.imag_std = 1.0

    def __len__(self):
        return len(self.masked_channels)

    def __getitem__(self, idx):
        # 获取输入和标签
        input_data = self.masked_channels[idx].copy()  # (M, K, 2)
        target_data = self.full_channels[idx].copy()  # (M, K, 2)

        # 归一化
        if self.normalize:
            input_data[:, :, 0] = (input_data[:, :, 0] - self.real_mean) / self.real_std
            input_data[:, :, 1] = (input_data[:, :, 1] - self.imag_mean) / self.imag_std
            target_data[:, :, 0] = (
                target_data[:, :, 0] - self.real_mean
            ) / self.real_std
            target_data[:, :, 1] = (
                target_data[:, :, 1] - self.imag_mean
            ) / self.imag_std

        input_tensor = torch.FloatTensor(input_data)
        target_tensor = torch.FloatTensor(target_data)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor

    def denormalize(self, tensor):
        """反归一化函数"""
        if not self.normalize:
            return tensor

        # tensor shape: (batch_size, 2, M, K) 或 (2, M, K)
        denorm_tensor = tensor.clone()
        denorm_tensor[:, 0] = (
            denorm_tensor[:, 0] * self.real_std + self.real_mean
        )  # 实部
        denorm_tensor[:, 1] = (
            denorm_tensor[:, 1] * self.imag_std + self.imag_mean
        )  # 虚部

        return denorm_tensor


def test_dataset_loading(mat_filename):
    """测试数据集加载"""
    print(f"测试加载文件: {mat_filename}")

    try:
        # 创建数据集
        dataset = FASUNetDataset(mat_filename, normalize=True)

        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 测试批次加载
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"\n批次 {batch_idx + 1}:")
            print(f"  输入形状: {inputs.shape}")  # (batch_size, 2, M, K)
            print(f"  目标形状: {targets.shape}")  # (batch_size, 2, M, K)
            print(f"  输入数据范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
            print(f"  目标数据范围: [{targets.min():.4f}, {targets.max():.4f}]")

            # 只测试第一个批次
            if batch_idx == 0:
                break

        print(f"\n✅ 数据集加载测试成功!")
        return dataset

    except Exception as e:
        print(f"❌ 数据集加载测试失败: {e}")
        import traceback

        traceback.print_exc()
        return None


# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    mat_file_path = "data/fas_unet_dataset.mat"

    # 测试加载
    dataset = test_dataset_loading(mat_file_path)

    if dataset is not None:
        print(f"\n数据集创建成功，包含 {len(dataset)} 个样本")

        # 可以进一步测试单个样本
        sample_input, sample_target = dataset[0]
        print(f"单个样本形状: 输入{sample_input.shape}, 目标{sample_target.shape}")
