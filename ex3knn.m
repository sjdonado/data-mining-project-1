clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Dataset given values
minRingsVal = 1;
maxRingsVal = 29;
totalRings = 1:maxRingsVal;
totalRings(28) = [];

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

% Defined H according to the project requeriments
H = eye(5, 9);

[N, n] = size(X);

% Find random abalone to plot
xrIdx = randi(N);
xr = X(xrIdx, :);
Xr = X(1:end ~= xrIdx,:);

xrh = xr*H';
Xh = Xr*H';

% 3-dimensions projection
[U, S, V] = svd(X,0);
U = U(:,1:3);
dS = diag(S);
pv = sum(dS(1:3))/sum(dS);

% Plot xr and its neighbors
figure;
hold all;
title(['KNN - PV = ', num2str(pv)]);
colors = spring(5);

ur = U(xrIdx,:);
Ur = U(1:end ~= xrIdx,:);
plot3(ur(1), ur(2), ur(3), 'ob', ...
    'markersize', 12, 'markerfacecolor', 'b');

% Defined k values for KNN search
k = [10 50 100 500 1000];

plotIdx = [];
for i = 1:5
    % Find neighborhood and estimate rings
    neighIdx = knnsearch(Xh, xrh, 'K', k(i));
    Xneigh = Xr(neighIdx,:);
    xrEsti = mode(Xneigh);

    estiRings = xrEsti(9);
    realRings = xr(9);
    success(i) = estiRings == realRings;

    plotIdx = setdiff(neighIdx, plotIdx);
    plot3(Ur(plotIdx, 1), Ur(plotIdx, 2), Ur(plotIdx, 3), 'ow', ...
        'markersize', 6, 'markerfacecolor', colors(i,:));
end

plot3(U(:, 1), U(:, 2), U(:, 3), 'ok', ...
    'markersize', 2, 'markerfacecolor', 'k');
legend(['Random abalone', string(k), 'All']');
grid on;

% Cross validation
NTra = floor(N * 0.7);
epochs = 100;

for k = [10 50 100 500 1000]
    % Training
    for it = 1:epochs
        % 70-30 random partition
        per = randperm(N);
        xPer = X(per, :);
        xTra = xPer(1:NTra, :);
        xVal = xPer(NTra + 1:end, :);
        [NVal, ~] = size(xVal);

        % Calculate nearest rings cluster by KNN search
        Xtra_atr = H * (xTra');
        Xval_atr = H * (xVal');

        values = knnsearch(Xtra_atr', Xval_atr', 'K', k);
        for idx = 1:NVal
            est(idx) = mode(values(idx, :)');
        end

        cat_est = xTra(est, 9); % Estimated
        cat_real = xVal(:, 9); % Actual

        A(it) = sum(cat_est == cat_real) / NVal; 
    end
    
    figure;
    hold all;
    title(['KNN - K = ', num2str(k)], 'fontsize', 20); 
    hist(A);
end

