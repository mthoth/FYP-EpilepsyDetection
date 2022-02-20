global t;
global bands;
global transformedData;
global fs;


M=8;

t = readtable('chb01_03.xlsx'); %replace file
t = t(:, 1:18);
t = table2array(t); %change type from table to array 
h = size(t, 1); %number of rows in 't'
n=(h/512)-2; 

transformedData=zeros(n, 434);
fs=256;
bands = linspace(0.5, 25, M + 1);

before=1;
after=512;
channels=18;
time = 6;
seizure_start = 2996;
seizure_stop = 3036;


[before, after]=getEpoch(before, after, 1, 1);
[before, after]=getEpoch(before, after, 1, 145);
[before, after]=getEpoch(before, after, 1, 289);
if time >= seizure_start && time <= seizure_stop
    transformedData(1, 433) = 1;
end
    transformedData(1, 433) = 0;
transformedData(1, 434) = time;
time = time + 2;
for row = 2:n % n = height(T) / number of rows per epoch
  transformedData(row, 1:288) = transformedData(row-1, 145:432); 
  [before, after]=getEpoch(before, after, row, 289);
  if time >= seizure_start && time <= seizure_stop
    transformedData(row, 433) = 1;
  else
    transformedData(row, 433) = 0;
  end
  transformedData(row, 434) = time;
  time = time + 2;
end