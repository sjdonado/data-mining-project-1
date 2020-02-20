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

% Correlation
C = corr(X);

figure;
imagesc(C);
colorbar;

% Defined H according to the project specifications
H = eye(3, 8);

% Calculate model error
X = [X Y];
[N, n] = size(X);
limiter = floor(N * 0.7);
epochs = 100;

for k = 1:epochs
    % 70-30 random partition
    per = randperm(N);
    x_per = X(per, :);
    x_tra = x_per(1:limiter, :);
    x_val = x_per(limiter + 1: N, :);
    
    % Calculate rings distribution moments by sex custlers
    for sex_idx = -1:1
        x_sex = x_tra(x_tra(:, 1) == sex_idx, 2:end);
        for rings_idx = 1:maxRingsVal
            x_r = x_sex(x_sex(:, end) == rings_idx, :);

            if ~isempty(x_r)
                [length, ~] = size(x_r);
                if length > 1
                    mean_r = H * (mean(x_r)');
                else
                    mean_r = H * (x_r');
                end
                m_rings{rings_idx} = mean_r;
                c_rings{rings_idx} = H * cov(x_r) * H';
            end 
        end
        s_m{sex_idx + 2} = m_rings;
        s_c{sex_idx + 2} = c_rings;
    end
    
    % Validation
    for idx = 1:N - limiter - 1
        sex_idx = x_val(idx, 1);
        xg = x_val(idx, 2:end);
        yg = H * (xg');
        cg = xg(8);
        
        m_rings_g = s_m{sex_idx + 2};
        c_rings_g = s_c{sex_idx + 2};
        
        % Adjust to nearest rings cluster
        for rings_idx = 1:maxRingsVal
            xb = m_rings_g{rings_idx}';
            B = c_rings_g{rings_idx};
            if ~isempty(xb)
               p_rings(rings_idx) = mvnpdf(yg, xb', B);
            end
        end
        
        [~, pos_rings] = max(p_rings);
        
        jejeje
    end
end