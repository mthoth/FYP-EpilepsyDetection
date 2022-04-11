import pandas as pd
import numpy as np

# ! Temporary global data. The data used in "fetchLiveData()" should start
# ! with the edf file uploaded by the user
preprocessedFileName = f'../Dataset/chb20/output/chb20_14_Preprocessed.csv'
preprocessedNdArr = np.asarray(pd.read_csv(preprocessedFileName))
preprocessedNdArr = preprocessedNdArr[930:1022, :]


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
    # TODO: Implementation. This should run predictions on uploaded edf from user and return...
    # x (time in seconds) and y (seizure labels (0/1))
    xCoordinates = preprocessedNdArr[:, 0]
    yCoordinates = preprocessedNdArr[:, -1]
    return xCoordinates, yCoordinates
