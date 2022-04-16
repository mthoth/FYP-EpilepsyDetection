import plotly.express as px
import pandas as pd
import numpy as np
import src.training2 as trn


preprocessedFileName = f'../Dataset/chb20/output/chb20_14_Preprocessed.csv'
preprocessedNdArr = np.asarray(pd.read_csv(preprocessedFileName))
preprocessedNdArr = preprocessedNdArr[980:992, :]


def runModel():
    import os

    try:
        print('Running model, please wait...')
        # ! To be changed depending on where you have the files on your machine
        PREPROCESSED_DIRECTORY = f'DashApp/data/preprocessed'
        preprocessedFiles = [file for file in os.listdir(
            PREPROCESSED_DIRECTORY) if file.endswith('.csv')]

        Xtrain, ytrain, Xtest, _ = trn.noisyData(
            PREPROCESSED_DIRECTORY, preprocessedFiles)
        fitModel = trn.SVM_CLASSWEIGHT_BALANCED(Xtrain, ytrain)
        yPred: np.ndarray = fitModel.predict(Xtest)

        # ! THIS SHOULD BE WHAT I WANT FOR GETTING VALUES NEAR THE FIRST SEIZURE OCCURENCE
        timingsAndLabels = np.array([(2*i + 6, label)
                                     for i, label in enumerate(yPred)])
        timings, labels = timingsAndLabels[:, 0], timingsAndLabels[:, 1]

        seizuresIndices = np.where(labels == 1)
        idxOfFirstSeizure = seizuresIndices[0][0]
        # idxSeizuresPred = [idx for idx,
        #                    label in enumerate(yPred) if label == 1]
        # # yTimings = [i*2 + 6 for i, label in enumerate(yPred)]

        # FOR TESTING
        return timings[idxOfFirstSeizure-10:idxOfFirstSeizure+10], labels[idxOfFirstSeizure-10:idxOfFirstSeizure+10]
        # BASE CASE
        # return timings, labels
    except Exception:
        print('Failed to train and test')


modelResults = runModel()


def fetchLiveData(dataIndex: int):
    idxStart, idxEnd = dataIndex, dataIndex + 10
    xCoords, yCoords = getPredictions()
    # x: time of the rows (epoch in seconds)
    times = [idx * 2 for idx, _ in enumerate(xCoords)]
    # y: label (0=non-ictal, 1=ictal in the corresponding epoch)
    labels = [seizureLabel for seizureLabel in yCoords]

    xCurrent, yCurrent = times[idxStart:idxEnd], labels[idxStart:idxEnd]
    return xCurrent, yCurrent


def getPredictions():
    # x (time in seconds) and y (seizure labels (0/1))
    xCoordinates, yCoordinates = modelResults

    # xCoordinates = preprocessedNdArr[:, 0]
    # yCoordinates = preprocessedNdArr[:, -1]
    return xCoordinates, yCoordinates


def getPatientStatus(figureY):
    try:
        return "DANGER" if figureY[-1] == 1 else "OK"
    except Exception:
        # ? When figureY[-1] is out of bounds
        return "???"


def getUpdatedGraph(currTime):
    figX, figY = fetchLiveData(currTime)
    fig = px.line(pd.DataFrame({
        'Time (s)': figX,
        'Seizure Label': figY
    }),
        x="Time (s)", y='Seizure Label', title=f'Patient Status: {getPatientStatus(figY)}', line_shape='hv', range_y=["-0.1", "1.1"])

    return fig, figX, figY
