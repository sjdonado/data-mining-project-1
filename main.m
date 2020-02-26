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
parsedSexVar(strcmpi(data.Sex, 'M')) = 1 * 10^-1;
parsedSexVar(strcmpi(data.Sex, 'I')) = 2 * 10^-1;
parsedSexVar(strcmpi(data.Sex, 'F')) = 3 * 10^-1;

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

% Correlation
C = corr(X);

figure;
imagesc(C);
colorbar;

% Defined H according to the project requeriments
H = eye(5, 9);

% Calculate model error
[N, n] = size(X);
limiter = floor(N * 0.7);
epochs = 100;

for k = 1:epochs
    % 70-30 random partition
    per = randperm(N);
    xPer = X(per, :);
    xTra = xPer(1:limiter, :);
    xVal = xPer(limiter + 1: N, :);
    [NVal, ~] = size(xVal);
    
    % Calculate moments by rings number
    for ringsIdx = totalRings
        xr = xTra(xTra(:, 9) == ringsIdx, :);
        [length, ~] = size(xr);
        if length > 1
            meanR = H * (mean(xr)');
        else
            meanR = H * (xr');
        end
        mRings{ringsIdx} = meanR;
        cRings{ringsIdx} = H * cov(xr) * H';
    end
    
    % Validation
    for idx = 1:NVal
        xg = xVal(idx, :);
        yg = H * (xg');
        cg = xg(9);
        
        % Adjust to nearest rings cluster
        for ringsIdx = totalRings
            xb = mRings{ringsIdx}';
            B = cRings{ringsIdx};
            [~, p] = chol(B);
            bisPositive = (p == 0 && rank(B) == size(B, 1));
            if ~isempty(xb) && bisPositive
               pRings(ringsIdx) = mvnpdf(yg, xb', B);
            end
        end
        
        [~, posRings] = max(pRings);

        estiRings(idx) = posRings;
        realRings(idx) = cg;
    end
    % Calculate success rate of the iteration
    A(k) = sum(estiRings == realRings) / (N - limiter)
end

fig = figure;
hist(A);
