%% 流体天线系统连续口径，时频网格矩阵G生成
%  编程人： 董学辉 
%  编程时间：2025年02月26日  
%  单位：HUST EIC
%  参考文献：Dong, Xuehui, et al. "Group Sparsity Methods for Compressive Space-Frequency
%           Channel Estimation and Spatial Equalization in Fluid Antenna System." arXiv preprint
%           arXiv:2503.02004 (2025).

%%   
clc; clear;
%% 参数初始化
omega_c  = 2*pi*5.8e9;          % 2pi*5.8GHz（角频率）
lambda_c = 2*pi*3e8/omega_c;    % 中心频点波长
B        = 2*pi*200e6;          % 带宽（角频率），1MHz*2pi
N_lambda = 10;
W        = N_lambda*lambda_c;   % 口径
K        = 128;                 % 频率/时延域频点网格点数
M        = 128;                 % 空间/角度域位置网格点数
%K        = 64;                 % 频率/时延域频点网格点数
%M        = 64;                 % 空间/角度域位置网格点数
w_k      = -B/2 : B/K : B/2 - B/K;

%% 生成网络以及感知矩阵
D_bold     = genOmega(K,M,B,W) * genPsi(K,M); %% 此函数输出维度较大（与M,K大小有关），耗时久，可保存输出矩阵直接调用
% 固定 N_r 和 N_p 的值
N_r        = 20;    % 固定天线数
N_p        = 40;    % 固定导频数
% 构造天线和导频位置采样矩阵（此处均采用 'static' 模式，可根据需要修改）
[S_Ir, Ir] = rowSamplingMatrix(N_r, M, 'static');
[S_Ip, Ip] = rowSamplingMatrix(N_p, K, 'random');
%S_Ip = rowSamplingMatrix(N_p, K, 'static');
% 感知矩阵 S 由 Kronecker 积构造
S          = kron(S_Ip, S_Ir);
% 生成测量矩阵
M_bold     = S * D_bold;
%% 频空网格 G 以及 观测向量 y 生成
% 参数设置
%rng('default');
rng(2);
L                = 90;                                 % 路径数量 L
tau_max          = 2e-7;                               % 路径最大时延 \tau_{max}
alpha_l_vec      = rand(1, L);                         % 路径增益 (随机生成示例值)
wavenumber_l_vec = rand(1, L) * 2 - 1;                 % 入射角 (均匀分布)---波数域的[-1，1)
tau_l_vec        = rand(1, L) * tau_max;               % 时延，单位: 秒 (随机生成示例值)
r                = linspace(0, W, M);        % 天线位置范围 单位: m
omega            = linspace(-B/2, B/2, K);             % 频率偏移范围 200MHz * 2 pi
tau              = linspace(0, tau_max, K);         % 时延域刻画（单位：秒），K 个采样点
wavenumber       = linspace(-1, 1, M);         % 波数域刻画（单位：归一化波数），M 个采样点
% 生成 G
G                = generateFSG(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r, omega);
g_0              = reshape(G,M*K,1);
% 生成 vec(Y_m)
Y_m_vec              = S*g_0;

save('frequency_space_grid_G.mat', 'G', 'g_0');                    % 保存G矩阵和向量化的g_0
save('observation_vector_Y.mat', 'Y_m_vec');                       % 保存观测向量
save('system_parameters.mat', 'K', 'M', 'L', 'B', 'W', 'omega_c', 'lambda_c', 'N_r', 'N_p'); % 保存系统参数




