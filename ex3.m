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

% Grouping one row clusters in pairs
ringsIdx = maxRingsVal;
while ringsIdx > 1
    xrIdxs = X(:, 9) == ringsIdx;
    xr = X(xrIdxs, :);
    if size(xr, 1) == 1
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
maxRingsVal = min(totalRings);

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
    
    % Calculate rings moments
    for ringsIdx = totalRings
        xr = xTra(xTra(:, 9) == ringsIdx, :);
        length = size(xr, 1);
        if length > 1
            lastRings{ringsIdx} = xr(end-1:end, :);
            mRings{ringsIdx} = H * (mean(xr)');
            cRings{ringsIdx} = H * cov(xr) * H';
        else
            if length == 1
                lastRings{ringsIdx} = xr;
                cRings{ringsIdx} = [];
            else
                lastRings{ringsIdx} = [];
            end
        end
    end
    
    % Fix moments calculations adding to the cluster the nearest neighborhoods 
    for ringsIdx = totalRings
        xr = xTra(xTra(:, 9) == ringsIdx, :);
     
        entry = isempty(lastRings{ringsIdx}) || isempty(cRings{ringsIdx});

        if ~entry
            B = cRings{ringsIdx};
            [~, p] = chol(B);
            bisPositive = (p == 0 && rank(B) == size(B, 1));
            entry = ~bisPositive;
        end

        if entry
            nxr = [];
            lastRings{ringsIdx} = [];
            
            if length > 0
                pIdx = ringsIdx - 1;
                aIdx = ringsIdx + 1;
                pRings = [];
                aRings = [];

                if pIdx >= minRingsVal
                    pRings = lastRings{pIdx};
                end
                
                if aIdx <= maxRingsVal
                    aRings = lastRings{aIdx};
                end
                
                nxr = [pRings; xr; aRings];
                
                length = size(nxr, 1);
                if length > 1
                    B = H * cov(nxr) * H';
                    [~, p] = chol(B);
                    bisPositive = (p == 0 && rank(B) == size(B, 1));
                    if bisPositive
                        lastRings{ringsIdx} = nxr(end-1:end, :);
                        mRings{ringsIdx} = H * (mean(nxr)');
                        cRings{ringsIdx} = B;
                    end
                end
            end
        end
    end

    % Cross validation
    for idx = 1:NVal
        xg = xVal(idx, :);
        yg = H * (xg');
        cg = xg(9);
        % Calculate nearest rings cluster
        for ringsIdx = totalRings
            if ~isempty(lastRings{ringsIdx})
                xb = mRings{ringsIdx}';
                B = cRings{ringsIdx};
                pRings(ringsIdx) = mvnpdf(yg, xb', B);
            end
        end
        
        [~, posRings] = max(pRings);
        estiRings(idx) = posRings;
        realRings(idx) = cg;
    end

    % Calculate success rate of the iteration
    A(k) = sum(estiRings == realRings) / NVal;
end

figure;
hist(A);
