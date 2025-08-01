function Omega = genOmega(K,M,B,W)
    % 生成网络以及感知矩阵
    w_k = -B/2:B/K:B/2-B/K; % 频率域的omega [-B/2,B/2)
    A   = cell(length(w_k),1);
    for k=1:length(w_k)
        A{k} = genArrayManifoldMatrix(M,W,w_k(k));
    end
    Omega = blkdiag(A{:}); %diag{A(w_1),A(w_2),...,A(w_K)}
end