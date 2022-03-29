global t;
global bands;
global transformedData;
global fs;


M=8;

t = xlsread('data/chb01_03.xlsx'); %replace file
t = t(:, 1:18);
t = table2array(t); %change type from table to array 
h = size(t, 1); %number of rows in 't'
n=(h/512)-2; 

transformedData=zeros(n, 433);
fs=256;
bands = linspace(0.5, 25, M + 1);

before=1;
after=512;
channels=18;
time = 6;
num_seizures = 1; %input the number of seizures
seizure_count = 1;
seizure_start = [2996]; %input the seizure start times in order
seizure_stop = [3036]; %input the seizure stop times in order


[before, after]=getEpoch(before, after, 1, 1);
[before, after]=getEpoch(before, after, 1, 145);
[before, after]=getEpoch(before, after, 1, 289);
if time > seizure_start(seizure_count) && time <= seizure_stop(seizure_count)
    transformedData(1, 433) = 1;
end
    transformedData(1, 433) = 0;
time = time + 2;
for row = 2:n % n = height(T) / number of rows per epoch
  if time > seizure_stop(seizure_count) && seizure_count < num_seizures
      seizure_count = seizure_count + 1;
  end
  transformedData(row, 1:288) = transformedData(row-1, 145:432); 
  [before, after]=getEpoch(before, after, row, 289);
  if time > seizure_start(seizure_count) && time <= seizure_stop(seizure_count)
      transformedData(row, 433) = 1;
  else
    transformedData(row, 433) = 0;
  end
  time = time + 2;
end