% 生成按SNR分组的FAS U-Net测试数据集
%
% 输入:
%   num_samples_per_snr (integer): 每个SNR值生成的样本数量
%   L_values (vector): L值数组 
%   output_filename (string): 保存数据集的 .mat 文件名
%
% 输出:
%   5维张量: (num_snr, num_samples_per_snr, M, K, 2)
%   其中第一维是SNR值，方便按SNR测试

num_samples_per_L = 150;
L = 20:20:160;
output_filename = 'C:\Users\dwyan\Desktop\data\unet_data\fas_train_dataset_by_L_snr10.mat';
SNR = [0 10];
%SNR = 10;
for L_idx = 1:length(L)
    for snr_idx = 1:length(SNR)
        output_filename = sprintf('C:\\Users\\dwyan\\Desktop\\data\\unet_data\\fas_train_dataset_by_L%d_snr%d.mat', L(L_idx),SNR(snr_idx));
        generate_fas_test_dataset_L(num_samples_per_L, L(L_idx),SNR(snr_idx), output_filename);
    end
end
