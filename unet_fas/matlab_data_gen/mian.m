%% 流体天线系统深度学习数据集生成（严格按照fas.m步骤）
%  编程人： 董学辉 
%  编程时间：2025年06月5日  
%  单位：HUST EIC
%  用途：深度学习信道估计数据集生成

clc; clear;

%% 数据集参数设置
num_samples = 10000;        % 训练样本数量
num_test = 2000;           % 测试样本数量

%% 参数初始化（完全按照fas.m）
omega_c  = 2*pi*5.8e9;          % 2pi*5.8GHz（角频率）
lambda_c = 2*pi*3e8/omega_c;    % 中心频点波长
B        = 2*pi*200e6;          % 带宽（角频率），200MHz*2pi
N_lambda = 10;
W        = N_lambda*lambda_c;   % 口径
K        = 128;                 % 频率/时延域频点网格点数
M        = 128;                 % 空间/角度域位置网格点数
w_k      = -B/2 : B/K : B/2 - B/K;

%% 生成网络以及感知矩阵（按照fas.m步骤）
fprintf('生成字典矩阵...\n');
D_bold     = genOmega(K,M,B,W) * genPsi(K,M); %% 此函数输出维度较大（与M,K大小有关），耗时久，可保存输出矩阵直接调用

% 固定 N_r 和 N_p 的值
N_r        = 20;    % 固定天线数
N_p        = 40;    % 固定导频数

% 构造天线和导频位置采样矩阵（此处均采用 'static' 模式，可根据需要修改）
[S_Ir, Ir] = rowSamplingMatrix(N_r, M, 'static');
[S_Ip, Ip] = rowSamplingMatrix(N_p, K, 'static');
% 感知矩阵 S 由 Kronecker 积构造
S          = kron(S_Ip, S_Ir);
% 生成测量矩阵
M_bold     = S * D_bold;

% 保存预计算的矩阵
%save('C:\Users\dwyan\Desktop\data\unet_data\dictionary_matrix.mat', 'D_bold', '-v7.3');
%save('C:\Users\dwyan\Desktop\data\unet_data\measurement_matrix.mat', 'M_bold', '-v7.3');
%save('C:\Users\dwyan\Desktop\data\unet_data\sampling_matrix.mat', 'S', 'S_Ir', 'S_Ip', 'Ir', 'Ip');

%% 生成训练集
fprintf('生成训练集 (%d 样本)...\n', num_samples);
%[train_data, train_labels] = generateDataset(num_samples, S, K, M, W, B,'train');
%save('C:\Users\dwyan\Desktop\data\unet_data\train_dataset.mat', 'train_data', 'train_labels', '-v7.3');

%% 生成测试集
fprintf('生成测试集 (%d 样本)...\n', num_test);
[test_data, test_labels] = generateDataset(num_test, S, K, M, W, B, 'test');
save('C:\Users\dwyan\Desktop\data\unet_data\test_dataset.mat', 'test_data', 'test_labels', '-v7.3');

%% 生成验证集（按SNR分层收集）
fprintf('生成验证集 (按SNR分层收集)...\n');

% 定义SNR范围和每个SNR的样本数
snr_levels = [0, 5, 10, 15, 20, 25, 30];  % SNR等级 (dB)
samples_per_snr = 200;  % 每个SNR等级收集的样本数
total_val_samples = length(snr_levels) * samples_per_snr;

fprintf('SNR等级: %s\n', mat2str(snr_levels));
fprintf('每个SNR样本数: %d\n', samples_per_snr);
fprintf('验证集总样本数: %d\n', total_val_samples);

[val_data, val_labels, val_snr_info] = generateDatasetBySNR(snr_levels, samples_per_snr, S, K, M, W, B);

% 保存分层验证集（单个文件，包含所有SNR数据）
save('val_dataset_stratified.mat', 'val_data', 'val_labels', 'val_snr_info', '-v7.3');

% 按SNR分别保存（方便Python按需加载）
saveValidationBySNR(val_data, val_labels, val_snr_info);

%% 保存数据集信息
dataset_info.num_samples = num_samples;
dataset_info.num_test = num_test;
dataset_info.num_val = total_val_samples;
dataset_info.K = K;
dataset_info.M = M;
dataset_info.N_r = N_r;
dataset_info.N_p = N_p;
dataset_info.input_dim = 2 * N_r * N_p;    % 实部虚部分离
dataset_info.output_dim = 2 * M * K;       % 实部虚部分离
dataset_info.omega_c = omega_c;
dataset_info.lambda_c = lambda_c;
dataset_info.B = B;
dataset_info.W = W;
% 添加验证集SNR信息
dataset_info.val_snr_levels = snr_levels;
dataset_info.samples_per_snr = samples_per_snr;
dataset_info.val_collection_strategy = 'stratified_by_snr';
save('dataset_info.mat', 'dataset_info');

% 检查数据完整性
checkValidationDataset();

fprintf('\n数据集生成完成!\n');
fprintf('训练集: %d 样本\n', num_samples);
fprintf('测试集: %d 样本\n', num_test);
fprintf('验证集: %d 样本 (按%d个SNR等级分层)\n', total_val_samples, length(snr_levels));

%% 数据集生成函数（严格按照fas.m的频空网格生成步骤）
function [data, labels] = generateDataset(num_samples, S, K, M, W, B, split_type)
    % 初始化存储
    input_dim = size(S, 1);         % N_r * N_p = 800
    output_dim = M * K;             % 128 * 128 = 16384
    
    data = zeros(2*input_dim, num_samples);      % 观测向量Y (输入)
    labels = zeros(2*output_dim, num_samples);   % 真实信道G (标签)
    
    % 路径数量和噪声范围（增加多样性）
    if strcmp(split_type, 'train')
        L_range = [50, 150];        % 训练集路径数变化范围
        snr_range = [0, 30];        % 信噪比范围 (dB)
    else
        L_range = [60, 120];        % 测试/验证集
        snr_range = [5, 25];
    end
    
    % 系统参数（按照fas.m）
    tau_max = 2e-7;                 % 路径最大时延
    
    for i = 1:num_samples
        if mod(i, 1000) == 0
            fprintf('  生成第 %d/%d 样本\n', i, num_samples);
        end
        
        %% 频空网格 G 以及 观测向量 y 生成（严格按照fas.m步骤）
        % 参数设置（每个样本随机化）
        L                = randi(L_range);                  % 路径数量 L
        alpha_l_vec      = rand(1, L);                      % 路径增益 (随机生成示例值)
        wavenumber_l_vec = rand(1, L) * 2 - 1;              % 入射角 (均匀分布)---波数域的[-1，1)
        tau_l_vec        = rand(1, L) * tau_max;            % 时延，单位: 秒 (随机生成示例值)
        
        % 按照fas.m的精确步骤
        r                = linspace(0, W, M);               % 天线位置范围 单位: m
        omega            = linspace(-B/2, B/2, K);          % 频率偏移范围 200MHz * 2 pi
        
        % 生成 G（使用fas.m中的generateFSG函数）
        G                = generateFSG(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r, omega);
        g_0              = reshape(G,M*K,1);
        
        % 生成 vec(Y_m)（按照fas.m步骤）
        Y_m_vec_clean    = S*g_0;
        
        % 添加噪声（深度学习数据集需要）
        snr_db = snr_range(1) + (snr_range(2) - snr_range(1)) * rand();
        signal_power = mean(abs(Y_m_vec_clean).^2);
        noise_power = signal_power / (10^(snr_db/10));
        noise = sqrt(noise_power/2) * (randn(size(Y_m_vec_clean)) + 1j*randn(size(Y_m_vec_clean)));
        Y_m_vec = Y_m_vec_clean + noise;
        
        % 存储数据（实部虚部分离）
        data(:, i) = [real(Y_m_vec); imag(Y_m_vec)];        % 观测向量（网络输入）
        labels(:, i) = [real(g_0); imag(g_0)];              % 真实信道（网络标签）
    end
end

%% 按SNR分层的数据集生成函数
function [data, labels, snr_info] = generateDatasetBySNR(snr_levels, samples_per_snr, S, K, M, W, B)
    % 初始化存储
    input_dim = size(S, 1);         % N_r * N_p = 800
    output_dim = M * K;             % 128 * 128 = 16384
    total_samples = length(snr_levels) * samples_per_snr;
    
    data = zeros(2*input_dim, total_samples);       % 观测向量Y (输入)
    labels = zeros(2*output_dim, total_samples);    % 真实信道G (标签)
    
    % SNR信息记录
    snr_info.snr_levels = snr_levels;
    snr_info.samples_per_snr = samples_per_snr;
    snr_info.sample_snr_labels = zeros(total_samples, 1);     % 每个样本对应的SNR
    snr_info.snr_start_indices = zeros(length(snr_levels), 1); % 每个SNR开始的索引
    snr_info.snr_end_indices = zeros(length(snr_levels), 1);   % 每个SNR结束的索引
    
    % 固定参数
    L_range = [60, 120];        % 路径数范围
    tau_max = 2e-7;             % 最大时延
    
    sample_idx = 1;
    
    % 对每个SNR等级生成数据
    for snr_idx = 1:length(snr_levels)
        current_snr = snr_levels(snr_idx);
        fprintf('  生成SNR=%d dB的数据 (%d/%d)\n', current_snr, snr_idx, length(snr_levels));
        
        % 记录当前SNR的起始和结束索引
        snr_info.snr_start_indices(snr_idx) = sample_idx;
        snr_info.snr_end_indices(snr_idx) = sample_idx + samples_per_snr - 1;
        
        for i = 1:samples_per_snr
            if mod(i, 50) == 0
                fprintf('    生成第 %d/%d 样本 (SNR=%d dB)\n', i, samples_per_snr, current_snr);
            end
            
            %% 生成信道参数（每个样本随机化）
            L                = randi(L_range);                  % 路径数量
            alpha_l_vec      = rand(1, L);                      % 路径增益
            wavenumber_l_vec = rand(1, L) * 2 - 1;              % 波数 [-1,1]
            tau_l_vec        = rand(1, L) * tau_max;            % 时延
            
            % 空间和频率网格
            r                = linspace(0, W, M);               % 天线位置
            omega            = linspace(-B/2, B/2, K);          % 频率偏移
            
            % 生成频空网格
            G                = generateFSG(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r, omega);
            g_0              = reshape(G, M*K, 1);
            
            % 生成观测（固定SNR）
            Y_m_vec_clean    = S * g_0;
            
            % 添加指定SNR的噪声
            signal_power = mean(abs(Y_m_vec_clean).^2);
            noise_power = signal_power / (10^(current_snr/10));
            noise = sqrt(noise_power/2) * (randn(size(Y_m_vec_clean)) + 1j*randn(size(Y_m_vec_clean)));
            Y_m_vec = Y_m_vec_clean + noise;
            
            % 存储数据（实部虚部分离）
            data(:, sample_idx) = [real(Y_m_vec); imag(Y_m_vec)];
            labels(:, sample_idx) = [real(g_0); imag(g_0)];
            snr_info.sample_snr_labels(sample_idx) = current_snr;
            
            sample_idx = sample_idx + 1;
        end
    end
    
    fprintf('按SNR分层的验证集生成完成!\n');
    
    % 打印统计信息
    fprintf('\n=== 验证集统计信息 ===\n');
    for snr_idx = 1:length(snr_levels)
        start_idx = snr_info.snr_start_indices(snr_idx);
        end_idx = snr_info.snr_end_indices(snr_idx);
        fprintf('SNR %2d dB: 样本 %4d - %4d (%d个样本)\n', ...
                snr_levels(snr_idx), start_idx, end_idx, samples_per_snr);
    end
end

%% 按SNR分别保存验证数据
function saveValidationBySNR(val_data, val_labels, val_snr_info)
    % 创建SNR子目录
    snr_dir = 'validation_by_snr';
    if ~exist(snr_dir, 'dir')
        mkdir(snr_dir);
    end
    
    snr_levels = val_snr_info.snr_levels;
    
    % 按SNR分别保存
    for snr_idx = 1:length(snr_levels)
        snr_level = snr_levels(snr_idx);
        start_idx = val_snr_info.snr_start_indices(snr_idx);
        end_idx = val_snr_info.snr_end_indices(snr_idx);
        
        % 提取当前SNR的数据
        current_data = val_data(:, start_idx:end_idx);
        current_labels = val_labels(:, start_idx:end_idx);
        current_snr = snr_level;
        num_samples = size(current_data, 2);
        
        % 保存为单独文件（Python友好的命名）
        filename = sprintf('val_snr_%02d.mat', snr_level);
        filepath = fullfile(snr_dir, filename);
        
        save(filepath, 'current_data', 'current_labels', 'current_snr', 'num_samples', '-v7.3');
        fprintf('保存 SNR %d dB 数据到: %s (%d个样本)\n', snr_level, filepath, num_samples);
    end
    
    % 保存SNR索引映射文件
    snr_index_file = fullfile(snr_dir, 'snr_index.mat');
    save(snr_index_file, 'val_snr_info', '-v7.3');
    
    fprintf('SNR分层验证数据保存完成到目录: %s\n', snr_dir);
end

%% 验证数据集完整性检查
function checkValidationDataset()
    fprintf('\n=== 验证数据集完整性检查 ===\n');
    
    % 检查总体验证集
    if exist('val_dataset_stratified.mat', 'file')
        load('val_dataset_stratified.mat', 'val_data', 'val_labels', 'val_snr_info');
        fprintf('✓ 总体验证集: %d个样本\n', size(val_data, 2));
        
        % 检查每个SNR的数据
        snr_dir = 'validation_by_snr';
        if exist(snr_dir, 'dir')
            snr_files = dir(fullfile(snr_dir, 'val_snr_*.mat'));
            fprintf('✓ SNR分层文件: %d个\n', length(snr_files));
            
            total_samples_check = 0;
            for i = 1:length(snr_files)
                file_path = fullfile(snr_dir, snr_files(i).name);
                data_info = load(file_path, 'current_snr', 'num_samples');
                fprintf('  SNR %d dB: %d个样本\n', data_info.current_snr, data_info.num_samples);
                total_samples_check = total_samples_check + data_info.num_samples;
            end
            
            if total_samples_check == size(val_data, 2)
                fprintf('✓ 样本数量一致性检查通过\n');
            else
                fprintf('✗ 样本数量不一致: 总体%d vs 分层%d\n', size(val_data, 2), total_samples_check);
            end
        else
            fprintf('✗ SNR分层目录不存在\n');
        end
    else
        fprintf('✗ 总体验证集文件不存在\n');
    end
end
