clc;
clear all;
close all;

data = readtable('abalone.data.csv');

% Rings statistics
minRingsVal = 1;
maxRingsVal = 29;

% Define the two colormaps for ring clusters.
cmap1 = hot(15);
cmap2 = winter(15);
% Combine them into one tall colormap.
combinedColorMap = [cmap1; cmap2];
randomRows = randi(size(combinedColorMap, 1), [maxRingsVal, 1]);
randomColors = combinedColorMap(randomRows, :);

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
% Get main diag for projection purposes
dS = diag(S);

% 2-dimensional space projection
pv = sum(dS(1:2))/sum(dS);

fig = figure;
hold all;

title(['PV = ', num2str(pv)], 'fontsize', 20);

for ringsNumber = 1:maxRingsVal
    idx = find(Y == ringsNumber);
    plot(U(idx, 1), U(idx, 2), 'ok', 'markersize', 8, ...
        'markerfacecolor', randomColors(ringsNumber, :));
end

le = legend(string(1:maxRingsVal-1)');
set(le, 'fontsize', 8, 'location', 'best');

grid on;

% 3-dimensional space projection
pv = sum(dS(1:3))/sum(dS);

fig = figure;
hold all;

title(['PV = ', num2str(pv)], 'fontsize', 20);

for ringsNumber = 1:maxRingsVal
    idx = find(Y == ringsNumber);
    plot3(U(idx, 1), U(idx, 2), U(idx, 3), 'ok', 'markersize', 8, ...
        'markerfacecolor', randomColors(ringsNumber, :));
end

le = legend(string(1:maxRingsVal-1)');
set(le, 'fontsize', 8, 'location', 'best');

grid on;
