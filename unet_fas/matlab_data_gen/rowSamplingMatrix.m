function [A,sortedChosenCols] = rowSamplingMatrix(m, n, mode)
    % 检查参数数量
    if nargin < 3
        error('必须提供三个参数：m, n, 和 mode');
    end
    
    % 检查 mode 是否有效
    if ~ischar(mode) || ~(strcmp(mode, 'static') || strcmp(mode, 'random') || strcmp(mode, 'randompilots'))
        error('mode 必须是 "static","randompilots" 或 "random"');
    end
    
    % 警告当 m 大于 n 时的情况
    if m > n
        warning('m 应该小于或等于 n，否则无法生成有效的矩阵');
    end
    
    % 初始化全零矩阵
    A = zeros(m, n);
    
    % 根据 mode 选择列索引
    if strcmp(mode, 'randompilots')
        % 随机模式：使用 randperm 生成随机列索引
        permCols = randperm(n);
        chosenCols = permCols(1:m);
        sortedChosenCols = sort(chosenCols);  % 对选择的列索引进行排序
        % 填充矩阵 A
        for i = 1:m
            A(i, sortedChosenCols(i)) = exp(1j*rand(1)*2*pi);
        end
    elseif strcmp(mode, 'random')
        permCols = randperm(n);
        chosenCols = permCols(1:m);
        sortedChosenCols = sort(chosenCols);  % 对选择的列索引进行排序
        % if ~any(sortedChosenCols==1, "all")
        %     sortedChosenCols(1)=1;
        % end
        % if ~any(sortedChosenCols==n, "all")
        %     sortedChosenCols(m)=n;
        % end
        % 填充矩阵 A
        for i = 1:m
            A(i, sortedChosenCols(i)) = 1;
        end
    else
        % 静态模式：使用 linspace 生成等间隔列索引
        intervals = linspace(1, n, m);
        sortedChosenCols = round(intervals);  % 四舍五入到最接近的整数
        % 填充矩阵 A
        for i = 1:m
            A(i, sortedChosenCols(i)) = 1;
        end
    end

end
