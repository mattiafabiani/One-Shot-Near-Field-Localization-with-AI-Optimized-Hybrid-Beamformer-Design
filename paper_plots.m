clc; clear; close all;

savefig = 0;
save_path = './imgs/results_matlab/';
fontsize = 2;
SNR_dB = 0:5:20;

%% Load Variables
csv_filename = 'test_rmse.csv';
model_path = 'saved_models/hyb_configs/';
DNN_hyb_fully_RF16 = readtable(strcat(model_path,'fully-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq_dnn/',csv_filename));
CNN_hyb_fully_RF16 = readtable(strcat(model_path,'fully-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq/',csv_filename));
CNN_hyb_inter_RF16 = readtable(strcat(model_path,'inter-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq/',csv_filename));
CNN_hyb_sub_RF16 = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq/',csv_filename));

% varying the number of RF CHAINS
csv_filename = 'test_rmse.csv';
model_path = 'saved_models/N_RF/';
CNN_hyb_sub_RF8_polar_uniform = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_8RF_128N_dropout01_eq/',csv_filename));
CNN_hyb_sub_RF16_polar_uniform = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq/',csv_filename));
CNN_hyb_sub_RF32_polar_uniform = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_32RF_128N_dropout01_eq/',csv_filename));
CNN_hyb_sub_RF64_polar_uniform = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_64RF_128N_dropout01_eq/',csv_filename));
% CNN_hyb_sub_RF128_polar_uniform = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_128RF_128N_dropout01_eq/',csv_filename));

% Multipath
csv_filename = 'test_rmse60.csv';
model_path = 'saved_models/multipath/';
% multipath_CNN_sub_N128_RF8 = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_8RF_128N_dropout01_eq_uniform/',csv_filename));
% multipath_CNN_sub_N128_RF16 = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_16RF_128N_dropout01_eq_uniform/',csv_filename));
% multipath_CNN_sub_N128_RF8 = readtable(strcat(model_path,'inter-connected_epochs50_batch256_lr0.001_8RF_128N_scat_CN/',csv_filename));
multipath_CNN_sub_N128_RF16 = readtable(strcat(model_path,'inter-connected_epochs50_batch256_lr0.001_16RF_128N_scat_CN/',csv_filename));
multipath_CNN_sub_N128_RF32 = readtable(strcat(model_path,'inter-connected_epochs50_batch256_lr0.001_32RF_128N_scat_CN/',csv_filename));
% multipath_CNN_sub_N128_RF32 = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_32RF_128N_scat_CN_random/',csv_filename));
% multipath_CNN_sub_N128_RF8_random = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_8RF_128N_scat_CN_random/',csv_filename));
% multipath_CNN_sub_N128_RF16_random = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_16RF_128N_scat_CN_random/',csv_filename));
% multipath_CNN_sub_N128_RF32_random = readtable(strcat(model_path,'sub-connected_epochs50_batch256_lr0.001_32RF_128N_scat_CN_random/',csv_filename));

% fully-digital (maximum likelihood)
model_path = 'saved_models/maximum_likelihood/';
% N = 128 antennas
maximum_likelihood_N128_rmse_r = readNPY(strcat(model_path,'rmse_r_N128_128_10.npy'));
maximum_likelihood_N128_rmse_theta_deg = readNPY(strcat(model_path,'rmse_theta_deg_N128_128_10.npy'));
maximum_likelihood_N128_rmse_theta_rad = readNPY(strcat(model_path,'rmse_theta_rad_N128_128_10.npy'));
maximum_likelihood_N128_rmse_pos = readNPY(strcat(model_path,'rmse_pos_N128_128_10.npy'));

%% Plotting
%%%%%%%%%%%%%%%%%%%%%% Compare different hybrid beamformers - and compare
%%%%%%%%%%%%%%%%%%%%%% with reference paper

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Figure settings
figure('Position', [100, 100, 1600, 400]);

% RMSE Angle Plot (subplot (a))
subplot(1, 3, 1);
plot(SNR_dB, CNN_hyb_inter_RF16.Test_theta_, '-og', 'DisplayName', 'inter-con', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16.Test_theta_, '-dr', 'DisplayName', 'sub-con', 'LineWidth', 1.5);
plot(SNR_dB, DNN_hyb_fully_RF16.Test_theta_, '--*b', 'DisplayName', 'fully-con [2]', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_fully_RF16.Test_theta_, '-*b', 'DisplayName', 'fully-con', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_theta_deg, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;

grid on;
xlabel('SNR [dB]', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\theta$ (deg)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
xlim([min(SNR_dB), max(SNR_dB)]);
set(gca, 'FontSize', 12); % Adjust axis font size

% RMSE Range Plot (subplot (b))
subplot(1, 3, 2);
plot(SNR_dB, CNN_hyb_inter_RF16.Test_r_, '-og', 'DisplayName', 'inter-con', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16.Test_r_, '-dr', 'DisplayName', 'sub-con', 'LineWidth', 1.5);
plot(SNR_dB, DNN_hyb_fully_RF16.Test_r_, '--*b', 'DisplayName', 'fully-con [2]', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_fully_RF16.Test_r_, '-*b', 'DisplayName', 'fully-con', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;

grid on;
xlabel('SNR [dB]', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $r$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
xlim([min(SNR_dB), max(SNR_dB)]);
set(gca, 'FontSize', 12); % Adjust axis font size

% RMSE Position Plot (subplot (c))
% Main plot (subplot 3)
subplot(1, 3, 3);
plot(SNR_dB, CNN_hyb_inter_RF16.Test_pos_, '-og', 'DisplayName', 'inter-con', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16.Test_pos_, '-dr', 'DisplayName', 'sub-con', 'LineWidth', 1.5);
plot(SNR_dB, DNN_hyb_fully_RF16.Test_pos_, '--*b', 'DisplayName', 'fully-con [2]', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_fully_RF16.Test_pos_, '-*b', 'DisplayName', 'fully-con', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_pos, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;

grid on;
xlabel('SNR [dB]', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\mathbf{p}$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
xlim([min(SNR_dB), max(SNR_dB)]);
set(gca, 'FontSize', 12); % Adjust axis font size

% Inset plot (magnifier)
% axes('Position', [0.75, 0.35, 0.1, 0.3]); % Adjust position and size of the inset
% box on;
% plot(SNR_dB, CNN_hyb_inter_RF16.Test_pos_, '-og', 'LineWidth', 1.5); hold on;
% plot(SNR_dB, CNN_hyb_sub_RF16.Test_pos_, '-dr', 'LineWidth', 1.5);
% plot(SNR_dB, DNN_hyb_fully_RF16.Test_pos_, '--*b', 'LineWidth', 1.5);
% plot(SNR_dB, CNN_hyb_fully_RF16.Test_pos_, '-*b', 'LineWidth', 1.5);
% plot(SNR_dB, maximum_likelihood_N128_rmse_pos, '-sk', 'LineWidth', 1.5); hold off;
% grid on;
% xlim([14.9, 15.4]); % Adjust this to zoom into the specific region of interest
% ylim([.7, 1.2]);   % Adjust the y-limits to the area of interest
% set(gca, 'FontSize', 10);
% set(gcf, 'Color', 'w');

if savefig
    saveas(gcf, strcat(save_path, 'hyb_configs'), 'epsc');
end


%%%%%%%%%%%%%%%%%%%%%% Results varying RF chains
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

figure('Position', [100, 100, 1600, 400]);

subplot(1, 3, 1);
plot(SNR_dB, CNN_hyb_sub_RF8_polar_uniform.Test_theta_, '-d', 'DisplayName', '$N_{RF}$ = 8', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16_polar_uniform.Test_theta_, '-o', 'DisplayName', '$N_{RF}$ = 16', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF32_polar_uniform.Test_theta_, '-v', 'DisplayName', '$N_{RF}$ = 32', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF64_polar_uniform.Test_theta_, '-*',  'DisplayName', '$N_{RF}$ = 64', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\theta$ (deg)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;

subplot(1, 3, 2);
plot(SNR_dB, CNN_hyb_sub_RF8_polar_uniform.Test_r_, '-d', 'DisplayName', '$N_{RF}$ = 8', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16_polar_uniform.Test_r_, '-o', 'DisplayName', '$N_{RF}$ = 16', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF32_polar_uniform.Test_r_, '-v', 'DisplayName', '$N_{RF}$ = 32', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF64_polar_uniform.Test_r_, '-*',  'DisplayName', '$N_{RF}$ = 64', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $r$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;

subplot(1, 3, 3);
plot(SNR_dB, CNN_hyb_sub_RF8_polar_uniform.Test_pos_, '-d', 'DisplayName', '$N_{RF}$ = 8', 'LineWidth', 1.5); hold on;
plot(SNR_dB, CNN_hyb_sub_RF16_polar_uniform.Test_pos_, '-o', 'DisplayName', '$N_{RF}$ = 16', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF32_polar_uniform.Test_pos_, '-v', 'DisplayName', '$N_{RF}$ = 32', 'LineWidth', 1.5);
plot(SNR_dB, CNN_hyb_sub_RF64_polar_uniform.Test_pos_, '-*',  'DisplayName', '$N_{RF}$ = 64', 'LineWidth', 1.5);
plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\mathbf{p}$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;
set(gcf, 'Color', 'w');

if savefig
    saveas(gcf, strcat(save_path, 'N_RF'), 'epsc');
end

%%%%%%%%%%%%%%%%%%%%%% Multipath - N_RF = {8,16}
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

figure('Position', [100, 100, 1600, 400]);

subplot(1, 3, 1);
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_theta_, '-or', 'DisplayName', '$N_{RF}$ = 8 direct path', 'LineWidth', 1.5); hold on;
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_theta_scat_, '--or', 'DisplayName', '$N_{RF}$ = 8 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_theta_, '-vb', 'DisplayName', '$N_{RF}$ = 16 direct path', 'LineWidth', 1.5); hold on;
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_theta_scat_, '--vb', 'DisplayName', '$N_{RF}$ = 16 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_theta_, '-or', 'DisplayName', '$N_{RF}$ = 32 direct path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_theta_scat_, '--or', 'DisplayName', '$N_{RF}$ = 32 2nd path', 'LineWidth', 1.5); hold off;
% plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5); hold off;
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\theta$ (deg)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;

subplot(1, 3, 2);
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_r_, '-or', 'DisplayName', '$N_{RF}$ = 8 direct path', 'LineWidth', 1.5); hold on;
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_r_scat_, '--or', 'DisplayName', '$N_{RF}$ = 8 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_r_, '-vb', 'DisplayName', '$N_{RF}$ = 16 direct path', 'LineWidth', 1.5); hold on;
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_r_scat_, '--vb', 'DisplayName', '$N_{RF}$ = 16 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_r_, '-or', 'DisplayName', '$N_{RF}$ = 32 direct path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_r_scat_, '--or', 'DisplayName', '$N_{RF}$ = 32 2nd path', 'LineWidth', 1.5); hold off;
% plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5);
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $r$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;

subplot(1, 3, 3);
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_pos_, '-or', 'DisplayName', '$N_{RF}$ = 8 direct path', 'LineWidth', 1.5); hold on;
% plot(SNR_dB, multipath_CNN_sub_N128_RF8.Test_pos_scat_, '--or', 'DisplayName', '$N_{RF}$ = 8 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_pos_, '-vb', 'DisplayName', '$N_{RF}$ = 16 direct path', 'LineWidth', 1.5); hold on;
plot(SNR_dB, multipath_CNN_sub_N128_RF16.Test_pos_scat_, '--vb', 'DisplayName', '$N_{RF}$ = 16 2nd path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_pos_, '-or', 'DisplayName', '$N_{RF}$ = 32 direct path', 'LineWidth', 1.5);
plot(SNR_dB, multipath_CNN_sub_N128_RF32.Test_pos_scat_, '--or', 'DisplayName', '$N_{RF}$ = 32 2nd path', 'LineWidth', 1.5); hold off;
% plot(SNR_dB, maximum_likelihood_N128_rmse_r, '-sk', 'DisplayName', 'fully-digital', 'LineWidth', 1.5);
xlim([min(SNR_dB), max(SNR_dB)]);
% ylim([0, y_lim_range]);
xticks(SNR_dB);
xlabel('SNR (dB)', 'FontSize', fontsize, 'Interpreter', 'latex');
ylabel('RMSE $\mathbf{p}$ (m)', 'FontSize', fontsize, 'Interpreter', 'latex');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust axis font size
grid on;
hold off;
set(gcf, 'Color', 'w');

if savefig
    saveas(gcf, strcat(save_path, 'multipath'), 'epsc');
end