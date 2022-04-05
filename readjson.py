import json
import os
import pandas as pd
from sympy import false
from preprocessing import Preprocessing

path = "data\chb01"
dir_list = os.scandir(path)


f = open('seizures.json')
json_data = json.load(f)

for list in dir_list:
        try:
            json_data[list[0:5]][list[6:8]]
        except:
            num = -1
            starts = [0]
            ends = [0]
            new_path = path + '\\' + list
            save_path = path + '\\output\\' + list[:-4] + "_Preprocessed.csv"
        else:
            num = json_data[list[0:5]][list[6:8]]['num']
            starts = json_data[list[0:5]][list[6:8]]['start']
            ends = json_data[list[0:5]][list[6:8]]['end']
            new_path = path + '\\' + list
            save_path = path + '\\output\\' + list[:-4] + "_Preprocessed.csv"
        
        print(list)
    # out = Preprocessing(new_path, num, starts, ends)

    # try:
    #     out.to_csv(save_path, index=False)
    # except:
    #     os.mkdir(path + '\\output\\')
    #     out.to_csv(save_path, index=False)

f.close()