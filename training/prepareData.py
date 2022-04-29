# from preprocessing.py import Preprocessing
from feature_selection import selectFeatures_ttest
import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from my_constants import PREPROCESSED_DIR


train_portion = 0.7
train_portion1 = 1
rand_state=1

def preparenoisydata(files): #data for training
    Final_X_train = np.empty([0, 432], dtype=object)
    Final_Y_train = np.empty(0)

    Final_X_test = np.empty([0, 432], dtype=object)
    Final_Y_test = np.empty(0)
    for file in files:
        curr_pd = pd.read_csv(F'{PREPROCESSED_DIR}/{file}')
        X = np.array(curr_pd.iloc[:, :-1])
        y = np.array(curr_pd.iloc[:, -1])

        if len(np.where(y == 1)[0]) != 0:
            Final_X_train, Final_Y_train, Final_X_test, Final_Y_test = dataseizure(X, y, Final_X_train, Final_Y_train, Final_X_test, Final_Y_test)
        else:
            Final_X_train, Final_Y_train, Final_X_test, Final_Y_test = nonseizure(X, y, Final_X_train, Final_Y_train, Final_X_test, Final_Y_test
             )

    return Final_X_train, Final_Y_train, Final_X_test, Final_Y_test

def preparecleandata(files):
    _, best_features = selectFeatures_ttest(files)
    print(f'Len {len(best_features)}')
    if "Outcome" not in best_features:
        best_features.append("Outcome")
    Final_X_train = np.empty([0, len(best_features) - 1], dtype=object)
    Final_Y_train = np.empty(0)

    Final_X_test = np.empty([0, len(best_features) - 1], dtype=object)
    Final_Y_test = np.empty(0)


    for file in files:
        curr_pd = pd.read_csv(f'{PREPROCESSED_DIR}/{file}', usecols=best_features)
        X = np.array(curr_pd.iloc[:, :-1])
        y = np.array(curr_pd.iloc[:, -1])

        if len(np.where(y == 1)[0]) != 0:
            Final_X_train, Final_Y_train, Final_X_test, Final_Y_test = dataseizure(X, y, Final_X_train, Final_Y_train,
                                                                                   Final_X_test, Final_Y_test)
        else:
            Final_X_train, Final_Y_train, Final_X_test, Final_Y_test = nonseizure(X, y, Final_X_train, Final_Y_train,
                                                                                  Final_X_test, Final_Y_test
                                                                                  )

    return Final_X_train, Final_Y_train, Final_X_test, Final_Y_test


def noisyData(files): #data for cross validation
    Final_X_train = np.empty([0, 432], dtype=object)
    Final_Y_train = np.empty(0)

    Final_X_test = np.empty([0, 432], dtype=object)
    Final_Y_test = np.empty(0)

    for file in files:
        curr_pd = pd.read_csv(f'{PREPROCESSED_DIR}/{file}')
        X = np.array(curr_pd.iloc[:, :-1])
        y = np.array(curr_pd.iloc[:, -1])

        
        if len(np.where(y == 1)[0]) != 0:
            Final_X_train, Final_Y_train= seizure(X, y, Final_X_train, Final_Y_train)
       # else:
        #    Final_X_train, Final_Y_train = nonseizure(X, y, Final_X_train, Final_Y_train
                                                                               #)
                
    return Final_X_train, Final_Y_train

def cleanData(files): #data for cross validation
    _, best_features = selectFeatures_ttest(files)
    print(f'Len {len(best_features)}')
    if "Outcome" not in best_features:
        best_features.append("Outcome")
    Final_X_train = np.empty([0, len(best_features)-1], dtype=object)
    Final_Y_train = np.empty(0)



    for file in files:
        # print(file)
        curr_pd = pd.read_csv(f'{PREPROCESSED_DIR}/{file}', usecols=best_features)
        X = np.array(curr_pd.iloc[:, :-1])
        y = np.array(curr_pd.iloc[:, -1])

        
        if len(np.where(y == 1)[0]) != 0:
            Final_X_train, Final_Y_train= seizure(X, y, Final_X_train, Final_Y_train)
        #else:
         #   Final_X_train, Final_Y_train = nonseizure(X, y, Final_X_train, Final_Y_train,
          #                                                                    )
    return Final_X_train, Final_Y_train

def seizure(X,Y,Final_X_train, Final_Y_train):
    X_seizure = X[Y[:] == 1]
    seizure_instances = np.ones(len(X_seizure))
    X_not_seizure = X[Y[:] == 0]
    non_seizure_instances = np.zeros(len(X_not_seizure))



    X_train_fit = np.concatenate((X_seizure[:], X_not_seizure[:len(X_seizure)*6]), axis=0)
    # print(X_train_fit.shape)
    y_train_fit = np.concatenate((seizure_instances[:], non_seizure_instances[:len(seizure_instances)*6]), axis=0)
    # print(y_train_fit.shape)

    Final_X_train = np.concatenate((Final_X_train[:], X_train_fit[:]), axis=0)
    Final_Y_train = np.concatenate((Final_Y_train[:], y_train_fit[:]), axis=0)



    return Final_X_train, Final_Y_train

def dataseizure(X,Y,Final_X_train, Final_Y_train, Final_X_test, Final_Y_test):
    X_seizure = X[Y[:] == 1]
    seizure_instances = np.ones(len(X_seizure))
    X_not_seizure = X[Y[:] == 0]
    non_seizure_instances = np.zeros(len(X_not_seizure))

    x_seizure_train, x_seizure_test, y_seizure_train, y_seizure_test = train_test_split(
        X_seizure,
        seizure_instances,
        train_size=train_portion,
        random_state=rand_state)

    x_nonseizure_train, x_nonseizure_test, y_nonseizure_train, y_nonseizure_test = train_test_split(
        X_not_seizure,
        non_seizure_instances,
        train_size=train_portion1,
        random_state=rand_state)

    X_train_fit = np.concatenate((x_seizure_train[:], x_nonseizure_train[:len(x_seizure_train) * 6]), axis=0)
    # print(X_train_fit.shape)
    y_train_fit = np.concatenate((y_seizure_train[:], y_nonseizure_train[:len(y_seizure_train) * 6]), axis=0)
    # print(y_train_fit.shape)

    X_test_fit = np.concatenate((x_seizure_test[:], x_nonseizure_test[:len(x_seizure_test) * 3]), axis=0)
    # print(X_test_fit.shape)
    y_test_fit = np.concatenate((y_seizure_test[:], y_nonseizure_test[:len(y_seizure_test) * 3]), axis=0)
    # print(y_test_fit.shape)
    # print(f'Shape of Final X {Final_X_train.shape}')
    # print(f'Shape of X train {X_train_fit.shape}')
    Final_X_train = np.concatenate((Final_X_train[:], X_train_fit[:]), axis=0)
    Final_Y_train = np.concatenate((Final_Y_train[:], y_train_fit[:]), axis=0)

    Final_X_test = np.concatenate((Final_X_test[:], X_test_fit[:]), axis=0)
    Final_Y_test = np.concatenate((Final_Y_test[:], y_test_fit[:]), axis=0)
    return Final_X_train, Final_Y_train, Final_X_test, Final_Y_test



def nonseizure(X,Y,Final_X_train, Final_Y_train, Final_X_test, Final_Y_test):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        train_size=train_portion,
        random_state=rand_state)
    Final_X_train = np.concatenate((Final_X_train[:], x_train[:2000]), axis=0)
    Final_Y_train = np.concatenate((Final_Y_train[:], y_train[:2000]), axis=0)
    Final_X_test = np.concatenate((Final_X_test[:], x_test[:100]), axis=0)
    Final_Y_test = np.concatenate((Final_Y_test[:], y_test[:100]), axis=0)



    return Final_X_train, Final_Y_train, Final_X_test, Final_Y_test

def data(files):
    Final_X = np.empty([0, 432], dtype=object)
    Final_Y = np.empty(0)

    for file in files:
        curr_pd = pd.read_csv(f'{PREPROCESSED_DIR}/{file}')
        X = np.array(curr_pd.iloc[:, :-1])
        y = np.array(curr_pd.iloc[:, -1])
        Final_X= np.concatenate((Final_X[:], X), axis=0)
        Final_Y = np.concatenate((Final_Y[:], y), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(Final_X, Final_Y, train_size=train_portion,
                                                                                random_state=rand_state)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    files = [file for file in os.listdir(PREPROCESSED_DIR) if '.csv' in file]