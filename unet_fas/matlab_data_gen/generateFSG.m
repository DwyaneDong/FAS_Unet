function G = generateFSG(L, alpha_l_vec, wavenumber_l_vec, tau_l_vec, r, omega)

    omega_c = 5.8 * pi * 2e9;      % 载波角频率，5.8GHz示例
    c       = 3e8;                     % 光速，单位: m/s
    M       = length(r);
    K       = length(omega);
    
    % 计算 离散化后的g(r, omega)，即FSG
    G_temp   = zeros(M, K);
    for l = 1:L
        for i = 1:length(r)
            for j = 1:length(omega)
                % 原始信号
                phase        = (omega(j) + omega_c) * ( r(i) * wavenumber_l_vec(l) + c * tau_l_vec(l) ) / c;
                G_temp(i, j) = G_temp(i, j) + alpha_l_vec(l) * exp(1j * phase);
            end
        end
    end
    G  = G_temp;
end