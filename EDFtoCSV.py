import os
import pandas as pd
import numpy as np
import pyedflib
import shutil

path = r'Dataset\chb01'
destination_path = r'Dataset\Dataset_Excelsheets\chb01'
files = os.listdir(path)
len_dir = len(files)
print(files)
for file in files:
    if '.txt' not in file:
        file_path = r'{}\{}'.format(path, file)
        data = pyedflib.EdfReader(file_path)
        data_df = pd.DataFrame()
        signals = data.getSignalLabels()
        signals = signals[:18]
        for i in range(18):
            data_df.insert(i, signals[i], data.readSignal(i))
        csv_name = file[:8] + '.csv'
        data_df.to_csv(csv_name, header=signals)
        print(csv_name)
        source_dir = r'{}'.format(csv_name)
        destination_dir = r'{}'.format(destination_path)
        shutil.move(source_dir, destination_dir)
