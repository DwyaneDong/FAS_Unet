function generate_fas_unet_datasetL(num_samples, output_filename)
    % 生成用于U-Net训练的流体天线信道数据集
    %
    % 输入:
    %   num_samples (integer): 要生成的样本数量
    %   output_filename (string): 保存数据集的 .mat 文件名
    %
    % 输出:
    %   在指定路径下保存一个 .mat 文件，包含:
    %     'all_masked_channels': cell array, 每个cell是 M x K x 2 的矩阵 (实部, 虚部) 作为模型输入
    %     'all_full_channels': cell array, 每个cell是 M x K x 2 的矩阵 (实部, 虚部) 作为模型标签
    %     'M_fixed', 'K_fixed': 网格维度

    if nargin < 2
        output_filename = 'fas_unet_dataset.mat';
    end
    if nargin < 1
        num_samples = 5000; % 默认生成1000个样本
    end

    fprintf('开始生成 %d 个样本，保存到 %s\n', num_samples, output_filename);

    % --- 基本固定参数 (可以根据需要调整) ---
    omega_c_base  = 2*pi*5.8e9;
    lambda_c_base = 2*pi*3e8/omega_c_base;
    B_base        = 2*pi*200e6;
    N_lambda_base = 10;
    W_base        = N_lambda_base*lambda_c_base;
    K_fixed       = 128; % 固定网格大小
    M_fixed       = 128;

    all_masked_channels = cell(num_samples, 1);
    all_full_channels   = cell(num_samples, 1);

    % 预计算 Psi (如果 D_bold 的 genOmega 部分不依赖于每次迭代变化的参数)
    Psi_fixed = genPsi_local(K_fixed, M_fixed);

    for i_sample = 1:num_samples
        if mod(i_sample, 50) == 0
            fprintf('正在生成样本 %d/%d\n', i_sample, num_samples);
        end

        % --- 1. 随机化信道参数 ---
        L = randi([10, 120]); % 路径数量
        tau_max = (0.8 + 1.4*rand()) * 1e-7; % 最大时延 (0.8e-7 to 2.2e-7)
        
        % 路径增益 (复数, 幅度和相位随机)
        alpha_magnitudes = 0.1 + 0.9*rand(1, L); % 幅度在 [0.1, 1]
        alpha_phases = 2*pi*rand(1, L);
        alpha_l_vec = alpha_magnitudes .* exp(1j * alpha_phases);
        
        wavenumber_l_vec = rand(1, L) * 2 - 1; % 波数 [-1, 1)
        tau_l_vec = rand(1, L) * tau_max; % 时延

        r_grid = linspace(0, W_base, M_fixed);
        omega_grid = linspace(-B_base/2, B_base/2, K_fixed); % 角频率偏移

        % --- 2. 生成完整信道 G_full ---
        G_full = generateFSG_local(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r_grid, omega_grid, omega_c_base);

        % --- 3. 随机化采样参数 ---
        % 采样比例可以调整
        N_r = randi([round(M_fixed*0.05), round(M_fixed*0.25)]); % 采样 5% 到 25% 的天线位置
        N_p = randi([round(K_fixed*0.05), round(K_fixed*0.25)]); % 采样 5% 到 25% 的频率点

        spatial_modes = {'static', 'random'};
        freq_modes = {'random', 'static'};
        
        S_Ir_mode = spatial_modes{randi(length(spatial_modes))};
        S_Ip_mode = freq_modes{randi(length(freq_modes))};

        [~, Ir_indices] = rowSamplingMatrix_local(N_r, M_fixed, S_Ir_mode);
        [~, Ip_indices] = rowSamplingMatrix_local(N_p, K_fixed, S_Ip_mode);
        
        % --- 4. 构建U-Net输入 (Masked G_full) ---
        % 使用0填充未采样位置，也可以考虑使用特定的标记值
        G_masked = zeros(M_fixed, K_fixed, 'like', 1j); % 初始化为复数0

        % 创建一个采样掩码，指示哪些点被采样
        sampling_mask_spatial = false(M_fixed, 1);
        sampling_mask_spatial(Ir_indices) = true;
        sampling_mask_freq = false(1, K_fixed);
        sampling_mask_freq(Ip_indices) = true;
        
        final_sampling_mask = sampling_mask_spatial & sampling_mask_freq; % M x K logical mask

        G_masked(final_sampling_mask) = G_full(final_sampling_mask);

        % 分离实部和虚部作为不同通道
        input_unet = zeros(M_fixed, K_fixed, 2);
        input_unet(:,:,1) = real(G_masked);
        input_unet(:,:,2) = imag(G_masked);
        all_masked_channels{i_sample} = input_unet;

        % --- 5. 构建U-Net输出 (Full G_full) ---
        output_unet = zeros(M_fixed, K_fixed, 2);
        output_unet(:,:,1) = real(G_full);
        output_unet(:,:,2) = imag(G_full);
        all_full_channels{i_sample} = output_unet;
    end

    fprintf('转换数据格式以提高Python兼容性...\n');

    % 转换cell array为4D数值数组: (num_samples, M, K, 2)
    all_masked_array = zeros(num_samples, M_fixed, K_fixed, 2);
    all_full_array = zeros(num_samples, M_fixed, K_fixed, 2);

    for i = 1:num_samples
        all_masked_array(i, :, :, :) = all_masked_channels{i};
        all_full_array(i, :, :, :) = all_full_channels{i};
    end

    % 保存数值数组版本（同时保留cell array版本以防需要）
    save(output_filename, 'all_masked_channels', 'all_full_channels', ...
        'all_masked_array', 'all_full_array', 'M_fixed', 'K_fixed', '-v7.3');
    end

function Psi = genPsi_local(K,M)
    k_temp = (0:K-1);
    t_temp = (0:K-1)';
    F_dft = (1/sqrt(K))*exp(-1j * 2 * pi * t_temp * k_temp / K);
    I_M = eye(M);
    Psi = kron(F_dft,I_M);
end

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
            sortedChosenCols = round(n/2); % Or 1, or n, depending on preference for single sample
            if sortedChosenCols == 0; sortedChosenCols = 1; end
        else
            intervals = linspace(1, n, m);
            sortedChosenCols = unique(round(intervals)); % unique to handle m > n or rounding issues
            % If unique results in fewer than m, adjust (this is a simple fix)
            while length(sortedChosenCols) < m && length(sortedChosenCols) < n
                missing_count = m - length(sortedChosenCols);
                potential_adds = setdiff(1:n, sortedChosenCols);
                if isempty(potential_adds) break; end
                additional_samples = randsample(potential_adds, min(missing_count, length(potential_adds)));
                sortedChosenCols = sort([sortedChosenCols, additional_samples']);
            end
            if length(sortedChosenCols) > m
                sortedChosenCols = sortedChosenCols(1:m); % Trim if oversampled
            end

        end
        for i = 1:length(sortedChosenCols) % Iterate over actual chosen cols
             if i <= m % Ensure we don't write out of bounds for A's first dim
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
    
    % Vectorized calculation for speed if possible, or keep loops
    % Loops are easier to understand initially:
    for l_idx = 1:L
        % Pre-calculate parts of the phase that don't depend on r(i) or omega(j)
        const_phase_part_l = omega_c_carrier * c * tau_l_vec(l_idx) / c; % This simplifies
        
        for i_idx = 1:M_dim % iterate over r
            r_val = r(i_idx);
            spatial_phase_part_l = omega_c_carrier * r_val * wavenumber_l_vec(l_idx) / c;
            
            for j_idx = 1:K_dim % iterate over omega (frequency offset)
                omega_val = omega(j_idx);
                
                % Full phase calculation
                phase = (omega_val * (r_val * wavenumber_l_vec(l_idx) + c * tau_l_vec(l_idx)) / c) + ...
                        spatial_phase_part_l + ...
                        const_phase_part_l; 
                % Alternative phase from original:
                % phase = (omega(j_idx) + omega_c_carrier) * ( r(i_idx) * wavenumber_l_vec(l_idx) + c * tau_l_vec(l_idx) ) / c;

                G_temp(i_idx, j_idx) = G_temp(i_idx, j_idx) + alpha_l_vec(l_idx) * exp(1j * phase);
            end
        end
    end
    G = G_temp;
end
