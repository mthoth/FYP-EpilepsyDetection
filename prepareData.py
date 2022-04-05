import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

train_portion = 0.8
rand_state = 1

def selectFeatures_ttest(files):
#   files = [file for file in os.listdir() if '.xlsx' in file  ]
  feature_val = {}
  df = pd.read_excel(r'C:\Users\yousef hamadeh\Senior Project\Data\{}'.format(files[0]), engine='openpyxl')
  min = np.array(df).shape[1]
  best_features = []
  for file in files:
    curr_pd_new = pd.read_excel(file, engine='openpyxl')
    outcome_col = np.array(curr_pd_new["Outcome"], dtype=int)
    if len(np.where(outcome_col == 1)[0]) != 0: 
      for title in list(curr_pd_new.columns.values):
        A = np.array(curr_pd_new[title])
        max_A = np.max(A)
        A = A / max_A
        statistics, p_value = stats.ttest_ind(a=A,
                                      b=outcome_col,
                                      equal_var=True)
        feature_val[title] = p_value
        alfa = 5 * (10 **(-3))
        temp_best_features = []
        for key, val in feature_val.items():
          if val <= alfa:
            temp_best_features.append(key)
      if min > len(temp_best_features):
        min = len(temp_best_features)
        best_features = temp_best_features
  return [min, best_features]

def prepareData(directory):
  files = [file for file in directory if '.xlsx' in file]
  min, best_features = selectFeatures_ttest(files)
  for file in files:
    curr_pd = pd.read_csv(file, usecols=best_features)

    X = np.array(curr_pd.drop(["Outcome"], axis=1))
    Y = np.array(curr_pd["Outcome"])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, train_size=train_portion,
                                                                                random_state=rand_state)
    X_seizure = x_train[y_train[:] == 1]
    seizure_instances = np.ones(len(X_seizure))
    X_not_seizure = x_train[y_train[:] == 0]
    non_seizure_instances = np.zeros(len(X_not_seizure))
    print(f'Number of non seizure in X seizure {len(seizure_instances[seizure_instances[:] == 0])}')
    print(f'Number of seizure in X not seizure {len(non_seizure_instances[non_seizure_instances[:] == 1])}')
    X_train_fit = np.concatenate((X_seizure[:], X_not_seizure[:100]), axis=0)
    print(X_train_fit.shape)
    y_train_fit = np.concatenate((seizure_instances[:], non_seizure_instances[:100]), axis=0)
    print(y_train_fit.shape)

    Final_X = np.concatenate((Final_X[:], X_train_fit[:]), axis=0)
    Final_Y = np.concatenate((Final_Y[:], y_train_fit[:]), axis=0)

if __name__ == '__main__':
    current_path = Path(r'C:\Users\yousef hamadeh\Senior Project\FYP-EpilepsyDetection')
    X, y = prepareData(os.listdir(current_path.parent))
    print(X.shape, y.shape)