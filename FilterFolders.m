myFolder = 'C:\Users\yousef hamadeh\Documents\MATLAB\chb01';
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.csv'); 
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    %check if this file has seizures

    %get output column
    %create function
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    % Now do whatever you want with this file name,
    xls_filename = strcat(baseFileName(1:8), '.xlsx');
    preprocessed_data = PreprocessFile(baseFileName);
    preprocessed_data = [preprocessed_data, %output column]
    writematrix(preprocessed_data, strcat(baseFileName(1:8), '.xlsx'))
end
%code taken from https://matlab.fandom.com/wiki/FAQ#How_can_I_process_a_sequence_of_files.3F