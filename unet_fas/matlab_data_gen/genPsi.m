function Psi=genPsi(K,M)

    k_temp = (0:K-1);                 % 角频率域采样点
    t_temp = (0:K-1)';                % 时延域采样点
    F   = (1/sqrt(K))*exp(-1j * 2 * pi * t_temp * k_temp / K);% 生成过采样 DFT 矩阵
    I_M = eye(M);
    Psi = kron(F,I_M);

end