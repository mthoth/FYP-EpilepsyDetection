import pandas as pd
import numpy as np


preprocessedFileName = f'../Dataset/chb20/output/chb20_14_Preprocessed.csv'
preprocessedNdArr = np.asarray(pd.read_csv(preprocessedFileName))
preprocessedNdArr = preprocessedNdArr[980:992, :]


# * Simulating live data fetching...
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
    # TODO: Implementation. This should run predictions test data from preprocessed files and return...
    # x (time in seconds) and y (seizure labels (0/1))
    xCoordinates = preprocessedNdArr[:, 0]
    yCoordinates = preprocessedNdArr[:, -1]
    return xCoordinates, yCoordinates
