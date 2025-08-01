function A_wk = genArrayManifoldMatrix(M,W,w_k) 
    w_c = 2*pi*5.8e9; % 常数 wc
    c   = 3e8;          % 光速（或其他常数）
    
    % 初始化矩阵及向量
    m_vector = 0:(M-1);        %[0,W),分成M份
    u_vector = 1:-2/M:-1+2/M;  %角度域的[0，pi）--cos()--> 波数域的[1,-1)
    A_index  = m_vector' * u_vector;
    A_wk_angle = ((w_k + w_c) * W * A_index)/ (c * M) ;
    A_wk     = exp(1j * A_wk_angle);
end