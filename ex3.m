clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Dataset given values
minRingsVal = 1;
maxRingsVal = 29;
totalRings = 1:maxRingsVal;

% Get random colors
combinedColorMap = [hot(15); winter(14)];
randomRows = randi(size(combinedColorMap, 1), [maxRingsVal, 1]);

% Set the plot opts
colors = combinedColorMap(randomRows, :);
labels = totalRings;
labels(28) = [];
labels = string(labels)';

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

% Grouping rings clusters with less than 3 rows
ringsIdx = maxRingsVal;
while ringsIdx > 1
    xrIdxs = X(:, 9) == ringsIdx;
    xr = X(xrIdxs, :);
    if size(xr, 1) < 3
        if ringsIdx - 1 == 28
            ringsIdx = ringsIdx - 1;
        end
        prevXrIdx = X(:, 9) == ringsIdx - 1;
        prevXr = X(prevXrIdx, :);
        X(xrIdxs | prevXrIdx, 9) = ceil((xr(1, 9) + prevXr(1, 9)) / 2);
        ringsIdx = ringsIdx - 1;
    end
    ringsIdx = ringsIdx - 1;
end
totalRings = unique(X(:, 9),'rows')';
minRingsVal = min(totalRings);
maxRingsVal = max(totalRings);

% Defined H according to the project requeriments
H = eye(8, 9);
Hcv = eye(5, 9);
Hpa = eye(5, 8);
meml = cell(1, maxRingsVal);
sig = 0.01;

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
    
    % Calculate rings moments
    for ringsIdx = totalRings
        mRings{ringsIdx} = [];
        cRings{ringsIdx} = [];
        xr = xTra(xTra(:, 9) == ringsIdx, :);
        length = size(xr, 1);
        if length > 1
            xb = H * (mean(xr)');
            B = H * cov(xr) * H';
            [~, p] = chol(B);
            bisPositive = (p == 0 && rank(B) == size(B, 1));
            if bisPositive
                mRings{ringsIdx} = xb;
                cRings{ringsIdx} = B;
            end
        end
    end

    % Cross validation
    for idx = 1:NVal
        xg = xVal(idx, :);
        yg = Hcv * (xg');
        cg = xg(9);
        % Calculate nearest rings cluster
        for ringsIdx = totalRings
            if ~isempty(mRings{ringsIdx})
                xb = mRings{ringsIdx}';
                B = cRings{ringsIdx};
                if isempty(meml{ringsIdx})
                    meml{ringsIdx} = (inv(B) + (1 / sig^2) * Hpa' * Hpa);
                    memr{ringsIdx} = B \ (xb');
                end
                xa = meml{ringsIdx} \ (memr{ringsIdx} + (1 / sig^2) * Hpa' * yg);
                pRings(ringsIdx) = mvnpdf(xa, xb', B);
            end
        end
        [~, posRings] = max(pRings);
        estiRings(idx) = totalRings(posRings);
        realRings(idx) = cg;
    end
    
    % Calculate success rate of the iteration
    A(k) = sum(estiRings == realRings) / NVal;
end

figure;
histogram(A);
