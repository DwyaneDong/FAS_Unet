function generate_fas_test_dataset(num_samples_per_snr, snr_values, output_filename)
    % 生成按SNR分组的FAS U-Net测试数据集
    %
    % 输入:
    %   num_samples_per_snr (integer): 每个SNR值生成的样本数量
    %   snr_values (vector): SNR值数组 (dB)
    %   output_filename (string): 保存数据集的 .mat 文件名
    %
    % 输出:
    %   5维张量: (num_snr, num_samples_per_snr, M, K, 2)
    %   其中第一维是SNR值，方便按SNR测试

    if nargin < 3
        output_filename = 'fas_test_dataset_by_snr.mat';
    end
    if nargin < 2
        snr_values = -10:5:20; % 默认SNR范围: -10dB to 20dB, 步长5dB
    end
    if nargin < 1
        num_samples_per_snr = 100; % 每个SNR值100个样本
    end

    num_snr = length(snr_values);
    total_samples = num_snr * num_samples_per_snr;
    
    fprintf('开始生成测试数据集:\n');
    fprintf('  SNR值: [%s] dB\n', num2str(snr_values));
    fprintf('  每个SNR样本数: %d\n', num_samples_per_snr);
    fprintf('  总样本数: %d\n', total_samples);
    fprintf('  保存到: %s\n', output_filename);

    % --- 基本固定参数 ---
    omega_c_base  = 2*pi*5.8e9;
    lambda_c_base = 2*pi*3e8/omega_c_base;
    B_base        = 2*pi*200e6;
    N_lambda_base = 10;
    W_base        = N_lambda_base*lambda_c_base;
    K_fixed       = 128;
    M_fixed       = 128;

    % 初始化5维张量: (num_snr, num_samples_per_snr, M, K, 2)
    all_masked_tensor = zeros(num_snr, num_samples_per_snr, M_fixed, K_fixed, 2);
    all_full_tensor = zeros(num_snr, num_samples_per_snr, M_fixed, K_fixed, 2);
    
    % 为兼容性也保存cell array格式
    all_masked_channels = cell(num_snr, num_samples_per_snr);
    all_full_channels = cell(num_snr, num_samples_per_snr);

    for snr_idx = 1:num_snr
        current_snr = snr_values(snr_idx);
        fprintf('\n处理SNR = %d dB (%d/%d)\n', current_snr, snr_idx, num_snr);
        
        for sample_idx = 1:num_samples_per_snr
            if mod(sample_idx, 20) == 0
                fprintf('  样本 %d/%d\n', sample_idx, num_samples_per_snr);
            end

            % --- 1. 随机化信道参数 ---
            L = randi([10, 120]);
            tau_max = (0.8 + 1.4*rand()) * 1e-7;
            
            alpha_magnitudes = 0.1 + 0.9*rand(1, L);
            alpha_phases = 2*pi*rand(1, L);
            alpha_l_vec = alpha_magnitudes .* exp(1j * alpha_phases);
            
            wavenumber_l_vec = rand(1, L) * 2 - 1;
            tau_l_vec = rand(1, L) * tau_max;

            r_grid = linspace(0, W_base, M_fixed);
            omega_grid = linspace(-B_base/2, B_base/2, K_fixed);

            % --- 2. 生成完整信道 G_full ---
            G_full = generateFSG_local(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r_grid, omega_grid, omega_c_base);

            % --- 3. 添加噪声 (基于SNR) ---
            G_full_noisy = addNoiseToChannel(G_full, current_snr);

            % --- 4. 随机化采样参数 ---
            N_r = randi([round(M_fixed*0.05), round(M_fixed*0.25)]);
            N_p = randi([round(K_fixed*0.05), round(K_fixed*0.25)]);

            spatial_modes = {'static', 'random'};
            freq_modes = {'random', 'static'};
            
            S_Ir_mode = spatial_modes{randi(length(spatial_modes))};
            S_Ip_mode = freq_modes{randi(length(freq_modes))};

            [~, Ir_indices] = rowSamplingMatrix_local(N_r, M_fixed, S_Ir_mode);
            [~, Ip_indices] = rowSamplingMatrix_local(N_p, K_fixed, S_Ip_mode);
            
            % --- 5. 构建U-Net输入 (Masked G_full_noisy) ---
            G_masked = zeros(M_fixed, K_fixed, 'like', 1j);

            sampling_mask_spatial = false(M_fixed, 1);
            sampling_mask_spatial(Ir_indices) = true;
            sampling_mask_freq = false(1, K_fixed);
            sampling_mask_freq(Ip_indices) = true;
            
            final_sampling_mask = sampling_mask_spatial & sampling_mask_freq;
            G_masked(final_sampling_mask) = G_full_noisy(final_sampling_mask);

            % 分离实部和虚部
            input_data = zeros(M_fixed, K_fixed, 2);
            input_data(:,:,1) = real(G_masked);
            input_data(:,:,2) = imag(G_masked);

            % --- 6. 构建U-Net目标 (Clean G_full) ---
            target_data = zeros(M_fixed, K_fixed, 2);
            target_data(:,:,1) = real(G_full);  % 使用无噪声的完整信道作为目标
            target_data(:,:,2) = imag(G_full);

            % 存储到张量
            all_masked_tensor(snr_idx, sample_idx, :, :, :) = input_data;
            all_full_tensor(snr_idx, sample_idx, :, :, :) = target_data;
            
            % 也存储到cell array (向后兼容)
            all_masked_channels{snr_idx, sample_idx} = input_data;
            all_full_channels{snr_idx, sample_idx} = target_data;
        end
    end

    fprintf('\n保存数据集...\n');
    
    % 保存多种格式
    save(output_filename, ...
        'all_masked_tensor', 'all_full_tensor', ...       % 5D张量格式
        'all_masked_channels', 'all_full_channels', ...   % Cell array格式
        'snr_values', 'num_samples_per_snr', ...          % SNR信息
        'M_fixed', 'K_fixed', '-v7.3');
    
    fprintf('测试数据集生成完成!\n');
    fprintf('张量形状: (%d, %d, %d, %d, %d)\n', size(all_masked_tensor));
end

function G_noisy = addNoiseToChannel(G_clean, snr_db)
    % 向信道矩阵添加复高斯白噪声
    %
    % 输入:
    %   G_clean: 清洁的信道矩阵
    %   snr_db: 信噪比 (dB)
    %
    % 输出:
    %   G_noisy: 添加噪声后的信道矩阵
    
    % 计算信号功率
    signal_power = mean(abs(G_clean(:)).^2);
    
    % 计算噪声功率
    snr_linear = 10^(snr_db/10);
    noise_power = signal_power / snr_linear;
    
    % 生成复高斯白噪声
    noise_real = sqrt(noise_power/2) * randn(size(G_clean));
    noise_imag = sqrt(noise_power/2) * randn(size(G_clean));
    noise = noise_real + 1j * noise_imag;
    
    % 添加噪声
    G_noisy = G_clean + noise;
end

% 复制必要的本地函数
function [A,sortedChosenCols] = rowSamplingMatrix_local(m, n, mode)
    if nargin < 3, error('必须提供三个参数：m, n, 和 mode'); end
    if ~ischar(mode) || ~(strcmp(mode, 'static') || strcmp(mode, 'random') || strcmp(mode, 'randompilots'))
        error('mode 必须是 "static","randompilots" 或 "random"');
    end
    if m > n, warning('m (%d) > n (%d), 将选择所有n个元素', m, n); m = n; end
    if m == 0, A = zeros(0,n); sortedChosenCols = []; return; end

    A = zeros(m, n);
    if strcmp(mode, 'randompilots')
        permCols = randperm(n);
        chosenCols = permCols(1:m);
        sortedChosenCols = sort(chosenCols);
        for i = 1:m
            A(i, sortedChosenCols(i)) = exp(1j*rand(1)*2*pi);
        end
    elseif strcmp(mode, 'random')
        permCols = randperm(n);
        chosenCols = permCols(1:m);
        sortedChosenCols = sort(chosenCols);
        for i = 1:m
            A(i, sortedChosenCols(i)) = 1;
        end
    else % static
        if m == 1
            sortedChosenCols = round(n/2);
            if sortedChosenCols == 0; sortedChosenCols = 1; end
        else
            intervals = linspace(1, n, m);
            sortedChosenCols = unique(round(intervals));
            while length(sortedChosenCols) < m && length(sortedChosenCols) < n
                missing_count = m - length(sortedChosenCols);
                potential_adds = setdiff(1:n, sortedChosenCols);
                if isempty(potential_adds) break; end
                additional_samples = randsample(potential_adds, min(missing_count, length(potential_adds)));
                sortedChosenCols = sort([sortedChosenCols, additional_samples']);
            end
            if length(sortedChosenCols) > m
                sortedChosenCols = sortedChosenCols(1:m);
            end
        end
        for i = 1:length(sortedChosenCols)
             if i <= m
                A(i, sortedChosenCols(i)) = 1;
             end
        end
        if length(sortedChosenCols) < m && m <= n
             warning('Static mode generated %d unique indices for m=%d. Matrix A might have fewer than m rows with 1s.', length(sortedChosenCols), m);
        end
    end
end

function G = generateFSG_local(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r, omega, omega_c_carrier)
    c = 3e8;
    M_dim = length(r);
    K_dim = length(omega);
    
    G_temp = complex(zeros(M_dim, K_dim));
    
    for l_idx = 1:L
        const_phase_part_l = omega_c_carrier * c * tau_l_vec(l_idx) / c;
        
        for i_idx = 1:M_dim
            r_val = r(i_idx);
            spatial_phase_part_l = omega_c_carrier * r_val * wavenumber_l_vec(l_idx) / c;
            
            for j_idx = 1:K_dim
                omega_val = omega(j_idx);
                
                phase = (omega_val * (r_val * wavenumber_l_vec(l_idx) + c * tau_l_vec(l_idx)) / c) + ...
                        spatial_phase_part_l + ...
                        const_phase_part_l; 

                G_temp(i_idx, j_idx) = G_temp(i_idx, j_idx) + alpha_l_vec(l_idx) * exp(1j * phase);
            end
        end
    end
    G = G_temp;
end