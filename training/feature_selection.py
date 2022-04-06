from scipy import stats
import os
import pandas as pd
import numpy as np

def selectFeatures_ttest(files):
  feature_val = {}
  # df = pd.read_excel(files[0])
  df = pd.read_csv(r'C:\Users\Dell\PycharmProjects\pythonProject\chb10preprocessed\{}'.format(files[0]))
  min = np.array(df).shape[1]
  best_features = []
  for f in files:
    file_path = r'C:\Users\Dell\PycharmProjects\pythonProject\chb10preprocessed\{}'.format(f)
    curr_pd_new = pd.read_csv(file_path)
    outcome_col = np.array(curr_pd_new["Outcome"])
    if len(np.where(outcome_col == 1)[0]) != 0: 
      for title in list(curr_pd_new.columns.values):
        A = np.array(curr_pd_new[title])
        max_A = np.max(A)
        A = A / max_A
        statistics, p_value = stats.ttest_ind(a=A,
                                      b=outcome_col,
                                      equal_var=True)
        # print(f'{title}: {p_value}')
        feature_val[title] = p_value
        alfa = 0.005
        temp_best_features = []
        for key, val in feature_val.items():
          if val <= alfa:
            temp_best_features.append(key)
      if min > len(temp_best_features):
        min = len(temp_best_features)
        best_features = temp_best_features
  return [min, best_features]

if __name__ == "__main__":
    files = [file for file in os.listdir(r'C:\Users\Dell\PycharmProjects\pythonProject\chb10preprocessed') if '.csv' in file]
    print(files)
    min_len, best_features = selectFeatures_ttest(files)
    print(f"Min length {min_len}")
    print(f"Best features {best_features}")