function [preprocessed_table] = PreprocessFile(table)
%PREPROCESSFILE Summary of this function goes here
%   Detailed explanation goes here
global t;
global bands
global transformedData;
global fs;
global M;
t = readtable(table);
t = t(:, 2:19); %removes unwated column (index of rows)
t = table2array(t); %change type from table to array 

h = height(t); %number of rows in 't'
n=h/1536; 

M=8;
transformedData=zeros(n, 432);
fs=256;
bands = linspace(0.5, 25, M + 1);

before=1;
after=512;
for row = 1:n % n = height(T) / number of rows per epoch
[before, after]=getFirstEpoch(before, after, row);
[before, after]=getSecondEpoch(before, after, row);
y=getThirdEpoch(before, after, row);
end
preprocessed_table = transformedData;
end

