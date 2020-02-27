clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Dataset given values
minRingsVal = 1;
maxRingsVal = 29;
totalRings = 1:maxRingsVal;
totalRings(28) = [];

% Get random colors
combinedColorMap = [hot(15); winter(14)];
randomRows = randi(size(combinedColorMap, 1), [maxRingsVal, 1]);

% Set the plot opts
colors = combinedColorMap(randomRows, :);
labels = string(totalRings)';

% Set table headers
variableNames = {'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', ...
    'Shucked weight', 'Viscera weight', 'Shell weight','Rings'};

for idx = 1:9
    data.Properties.VariableNames{strcat('Var', num2str(idx))} = variableNames{idx};
end

% Convert table to matrix
matrixData = table2array(data(:, 2:9));

% Relabel Sex variable
parsedSexVar = zeros(4177, 1);
parsedSexVar(strcmpi(data.Sex, 'M')) = -1;
parsedSexVar(strcmpi(data.Sex, 'I')) = 0;
parsedSexVar(strcmpi(data.Sex, 'F')) = 1;

X = [parsedSexVar matrixData];

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

for ringsIdx = totalRings
    indexes = find(X(:, end) == ringsIdx);
    plot(U(indexes, 1), U(indexes, 2), 'ok', 'markersize', 8, ...
        'markerfacecolor', colors(ringsIdx, :));
end

le = legend(labels);
set(le, 'fontsize', 8, 'location', 'best');
grid on;

% 3-dimensional space projection
pv = sum(dS(1:3))/sum(dS);

figure;
hold all;
title(['PV = ', num2str(pv)], 'fontsize', 20);

for ringsIdx = totalRings
    indexes = find(X(:, end) == ringsIdx);
    plot3(U(indexes, 1), U(indexes, 2), U(indexes, 3), 'ok', ...
        'markersize', 8, 'markerfacecolor', colors(ringsIdx, :));
end

le = legend(labels);
set(le, 'fontsize', 8, 'location', 'best');

grid on;

% Print correlation
C = corr(X);
figure;
imagesc(C);
colorbar;

% Defined H according to the project requeriments
H = eye(5, 9);

[N, n] = size(X);
NTra = floor(N * 0.7);
epochs = 100;

% Training
for k = 1:epochs
    % 70-30 random partition
    per = randperm(N);
    xPer = X(per, :);
    xTra = xPer(1:NTra, :);
    xVal = xPer(NTra + 1:end, :);
    [NVal, ~] = size(xVal);

    % Calculate nearest rings cluster by KNN search
    Xtra_atr = H * (xTra');
    Xval_atr = H * (xVal');

    values = knnsearch(Xtra_atr', Xval_atr', 'K', 10);
    for idx = 1:xVal
        est(idx) = mode(values(idx, :)');
    end

    cat_est = xTra(est, 9); % Estimated
    cat_real = xVal(:, 9); % Actual
    
    A(k) = sum(cat_est == cat_real) / NVal; 
end

fig = figure;
hist(A);
