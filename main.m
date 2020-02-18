clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Dataset given values
minRingsVal = 1;
maxRingsVal = 29;

% Define the two colormaps for ring clusters.
cmap1 = hot(15);
cmap2 = winter(15);
% Combine them into one tall colormap.
combinedColorMap = [cmap1; cmap2];
randomRows = randi(size(combinedColorMap, 1), [maxRingsVal, 1]);

% Set the plot opts
colors = combinedColorMap(randomRows, :);
labels = string(1:maxRingsVal - 1)';

% Set table headers
variable_names = {'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', ...
    'Shucked weight', 'Viscera weight', 'Shell weight','Rings'};

for idx = 1:9
    data.Properties.VariableNames{strcat('Var', num2str(idx))} = variable_names{idx};
end

% Convert table to matrix
matrixData = table2array(data(:, 2:9));

% Relabel Sex variable
parsedSexVar = zeros(4177, 1);
parsedSexVar(strcmpi(data.Sex, 'M')) = -1;
parsedSexVar(strcmpi(data.Sex, 'I')) = 0;
parsedSexVar(strcmpi(data.Sex, 'F')) = 1;

matrixData = [parsedSexVar matrixData];

% Inputs variables
X = matrixData(:, 1:8);

% Rings
Y = matrixData(:, end);

% Singular values descomposition
[U, S, V] = svd(X);

PHI = V/S;
PHI3 = PHI(:, 1:3); %To project onto three dimensions
dS = diag(S);

% 2-dimensional space projection
pv = sum(dS(1:2))/sum(dS);

figure;
hold all;
title(['PV = ', num2str(pv)], 'fontsize', 20);

for ringsNumber = 1:maxRingsVal
    indexes = find(Y == ringsNumber);
    plot(U(indexes, 1), U(indexes, 2), 'ok', 'markersize', 8, ...
        'markerfacecolor', colors(ringsNumber, :));
end

le = legend(labels);
set(le, 'fontsize', 8, 'location', 'best');
grid on;

% 3-dimensional space projection
pv = sum(dS(1:3))/sum(dS);

figure;
hold all;
title(['PV = ', num2str(pv)], 'fontsize', 20);

for ringsNumber = 1:maxRingsVal
    indexes = find(Y == ringsNumber);
    plot3(U(indexes, 1), U(indexes, 2), U(indexes, 3), 'ok', ...
        'markersize', 8, 'markerfacecolor', colors(ringsNumber, :));
end

le = legend(labels);
set(le, 'fontsize', 8, 'location', 'best');

grid on;

% Projection by sex variable
figure;
hold all;
C = {'b','r','m'};
for la = -1:1
    indexes = find(X(:,1) == la);
    plot3(U(indexes, 1) , U(indexes, 2), U(indexes, 3), 'ok', ...
        'markersize', 8, 'markerfacecolor', C{la + 2});
end

le = legend('M', 'I', 'F');
set(le, 'fontsize', 8, 'location', 'best');
grid on;

% Correlation
C = corr(X);

figure;
imagesc(C);
colorbar;

% Defined H according to the project specifications
H = [1 0 0 0 0 0 0;
     0 1 0 0 0 0 0;
     0 0 1 0 0 0 0;
     0 0 0 1 0 0 0];

% Male
indexes = find(X(:,1) == -1);
US = U(indexes, [1 2 3]);
XSO = X(indexes, 2:end);
[N, n] = size(XSO);

limiter = floor(N * 0.7);

for k = 1:100
    per = randperm(N);
    XSO = XSO(per, :);
    XS = XSO(1:limiter, :);
    Xe = XSO(1:limiter, :);

    % Mean and Covariance (FIT)
    xb = mean(XS);
    B  = cov(XS);

    % Estimation
    sig = 0.01;
    eobs = sig * randn(4, 1);
    y = H*(Xe') + eobs;
    % Posterior
    xa = (inv(B) + (1 / sig^2) * H'* H) \ (B \ (xb') + (1 / sig^2) * H' * y);
    % Covarianza posterior
    Ca = inv(inv(B) + (1 / sig^2) * H' * H);

    est9 = xa(5:8);
    %var3 = Ca(3,3);
    ERR(k) = norm(est9 - Xe(5:8)', 1) / norm(Xe(5:8), 1);
end

hist(ERR)


