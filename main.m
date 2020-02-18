clc;
clear all;
close all;

data = readtable('abalone.data.csv');

variable_names = {'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', ...
    'Shucked weight', 'Viscera weight', 'Shell weight','Rings'};

% Set variable names
for idx = 1:9
    data.Properties.VariableNames{strcat('Var', num2str(idx))} = variable_names{idx};
end

% Relabel Sex variable
data.Sex(strcmpi(data.Sex, 'M')) = {-1};
data.Sex(strcmpi(data.Sex, 'I')) = {0};
data.Sex(strcmpi(data.Sex, 'F')) = {1};

% Inputs variables
X = data(:, 1:8);

% Rings
Y = data(:, end);


