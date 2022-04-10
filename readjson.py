import json
import os
import pandas as pd
from preprocessing import Preprocessing
from training.myConstants import PATIENT_DIR

path = PATIENT_DIR
dir_list = os.scandir(path)


f = open('seizures.json')
json_data = json.load(f)

for entry in dir_list:
    entry = entry.name

    try:
        json_data[entry[0:5]][entry[6:8]]
    except:
        num = -1
        starts = [0]
        ends = [0]
        new_path = path + '\\' + entry
        save_path = path + '\\output\\' + entry[:-4] + "_Preprocessed.csv"
    else:
        num = json_data[entry[0:5]][entry[6:8]]['num']
        starts = json_data[entry[0:5]][entry[6:8]]['start']
        ends = json_data[entry[0:5]][entry[6:8]]['end']
        new_path = path + '\\' + entry
        save_path = path + '\\output\\' + entry[:-4] + "_Preprocessed.csv"

    out = Preprocessing(new_path, num, starts, ends)
    if isinstance(out, pd.DataFrame):
        try:
            out.to_csv(save_path, index=False)
        except:
            os.mkdir(path + '\\output\\')
            out.to_csv(save_path, index=False)

f.close()
