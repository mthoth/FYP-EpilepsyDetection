import mne
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import src.training2 as trn


preprocessedFileName = f'DashApp/data/preprocessed/chb20_14_Preprocessed.csv'
preprocessedNdArr = np.asarray(pd.read_csv(preprocessedFileName))
preprocessedNdArr = preprocessedNdArr[960:1020, :]

# figData = None

globalModel = joblib.load('chb20model.pkl')

def runModel():
    import os

    try:
        print('Running model, please wait...')
        #! To be changed depending on where you have the files on your machine
        PREPROCESSED_DIRECTORY = f'DashApp/data/preprocessed'

        preprocessedFiles = [file for file in os.listdir(
            PREPROCESSED_DIRECTORY) if file.endswith('.csv')]

        Xtrain, ytrain, Xtest, _ = trn.noisyData(
            PREPROCESSED_DIRECTORY, preprocessedFiles)
        fitModel = trn.SVM_CLASSWEIGHT_BALANCED(Xtrain, ytrain)
        yPred: np.ndarray = fitModel.predict(Xtest)

        timingsAndLabels = np.array([(2*i + 6, label)
                                     for i, label in enumerate(yPred)])
        timings, labels = timingsAndLabels[:, 0], timingsAndLabels[:, 1]

        seizuresIndices = np.where(labels == 1)
        idxOfFirstSeizure = seizuresIndices[0][0]

        # FOR TESTING
        return timings[idxOfFirstSeizure-10:idxOfFirstSeizure+10], labels[idxOfFirstSeizure-10:idxOfFirstSeizure+10]

        # BASE CASE
        # return timings, labels
    except Exception:
        print('Failed to train and test')


def fetchLiveData(n: int):
    idxStart, idxEnd = n, n + 5
    xCoords, yCoords = getPredictions()
    # x: time of the rows (epoch in seconds)
    times = [idx * 2 for idx, _ in enumerate(xCoords)]
    # y: label (0=non-ictal, 1=ictal in the corresponding epoch)
    labels = [seizureLabel for seizureLabel in yCoords]

    xCurrent, yCurrent = times[idxStart:idxEnd], labels[idxStart:idxEnd]
    return xCurrent, yCurrent


def getPredictions():
    xCoordinates = preprocessedNdArr[:, 0]
    yCoordinates = preprocessedNdArr[:, -1]
    return xCoordinates, yCoordinates

def getPredictions2(n):
    global globalModel
    currentEpochPrediction = globalModel.predict(preprocessedNdArr[n:n+5, :-1])

    return {
        'x': [(n+idx) * 2 for idx, _ in enumerate(currentEpochPrediction)], 
        'y': currentEpochPrediction
    }
    

def getUpdatedGraph(currTime):
    def getPatientStatus(figureY):
        try:
            return "DANGER" if figureY[-1] == 1 else "OK"
        except Exception:
            # ? When figureY[-1] is out of bounds
            return "???"
    
    figX, figY = fetchLiveData(currTime)
    fig = px.line(pd.DataFrame({
        'Time (s)': figX,
        'Seizure Label': figY
    }),
        x="Time (s)", y='Seizure Label', title=f'Patient Status: {getPatientStatus(figY)}', line_shape='hv', range_y=["-0.1", "1.1"])

    return fig, figX, figY



def initEegData():
    edfFilename = '../Dataset/chb20/chb20_14.edf'
    
    print(f'############## Creating EEG Figure ##############')
    raw = mne.io.read_raw_edf(edfFilename, preload=True, verbose=False)
    return raw


def getActualY(time: int):
    rowIdx = time // 2
    return int(preprocessedNdArr[rowIdx, -1])