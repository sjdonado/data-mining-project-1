clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Dataset given values
minRingsVal = 1;
maxRingsVal = 29;

% Set the plot opts
totalRings = 1:maxRingsVal;
totalRings(28) = [];
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
    plot(U(indexes, 1), U(indexes, 2), '.', 'markersize', 20);
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
    plot3(U(indexes, 1), U(indexes, 2), U(indexes, 3), '.', ...
        'markersize', 20);
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
H = eye(4,8);

% Calculate model error
[N, n] = size(X);
limiter = floor(N * 0.7);
epochs = 10;

for k = 1:epochs
    % 70-30 random partition
    per = randperm(N);
    xPer = X(per, :);
    xTra = xPer(1:limiter, :);
    xVal = xPer(limiter + 1: N, :);
    
    % Calculate rings distribution moments by sex custlers
    for sexIdx = -1:1
        xSex = xTra(xTra(:, 1) == sexIdx, 2:end);
        for ringsIdx = totalRings
            xr = xSex(xSex(:, end) == ringsIdx, :);
            [length, ~] = size(xr);
            if length > 1
                meanR = H * (mean(xr)');
            else
                meanR = H * (xr');
            end
            mRings{ringsIdx} = meanR;
            cRings{ringsIdx} = H * cov(xr) * H';
        end
        meansBySex{sexIdx + 2} = mRings;
        covariancesBySex{sexIdx + 2} = cRings;
    end
    
    % Validation
    for idx = 1:N - limiter - 1
        sexIdx = xVal(idx, 1);
        xg = xVal(idx, 2:end);
        yg = H * (xg');
        cg = xg(8);
        
        mRingsG = meansBySex{sexIdx + 2};
        cRingsG = covariancesBySex{sexIdx + 2};
        
        % Adjust to nearest rings cluster
        for ringsIdx = totalRings
            xb = mRingsG{ringsIdx}';
            B = cRingsG{ringsIdx};
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
