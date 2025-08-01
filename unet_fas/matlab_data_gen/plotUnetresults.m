L = 20:20:160;
results= [0.725143,0.753974,0.769053,0.752310,0.773182,0.759881,0.768677,0.762224;
           0.701888,0.699447,  0.709280,0.726286,0.707248,0.731542,0.735607,0.732432];


% 创建一个新图像窗口
figure(5);
% 绘制各方法的误差曲线
plot(L , results(1,:), '-o', 'LineWidth', 1.2, 'MarkerSize', 4, 'Color', '#888888'); hold on;

plot(L , results(2,:), '--o', 'LineWidth', 1.2, 'MarkerSize', 4, 'Color', '#888888'); hold on;


% 设置图形属性
grid minor;
xlabel('Number of Path ($L$)', 'FontSize', 12,'Interpreter','latex');
ylabel('Relative Error', 'FontSize', 12,'Interpreter','latex');
set(gca, 'TickLabelInterpreter', 'latex');
title('Number of Iterations: 50','Interpreter','latex');
ylim([0.2 0.9]);
% 添加图例，位置可根据实际情况调整
legend('Unet, SNR=0dB','Unet, SNR=10dB', 'Location', 'west','Interpreter','latex','box','off');