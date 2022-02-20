function [before, after] = getEpoch(before, after, row, index)
%GETTHIRDEPOCH Summary of this function goes here
%   Detailed explanation goes here
global transformedData;
global fs;
global bands;
global t;
M=8;

for channel=1:18
x=t(before:after, channel);
for j = 1:M
transformedData(row, index) = bandpower(x, fs, bands([j j+1])); %row, index (index of a bandpower in matrix y)
index=index + 1;
end
end

before=after+1;
after=before+511;
end
