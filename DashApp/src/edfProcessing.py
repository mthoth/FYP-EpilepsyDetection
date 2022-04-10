import pandas as pd
import numpy as np
import mne
from mne.io.edf import edf

# from plotly.graph_objects import Scatter, Figure

# fileName = 'data/chb21_22.edf'
# raw = mne.io.read_raw_edf(fileName, verbose=False, preload=True)
# print(f'eeg file: "{fileName}" sucessfully loaded. ')

# start, stop = raw.time_as_index([0, 100])

# data, times = raw[:, start:stop]
# traces = [Scatter(x=times, y=data.T[:, 0])]

# eegFig = Figure(data=traces[:])


def readEdfAndConvertToDataframe(filename: str) -> np.ndarray:
    try:
        rawData = mne.io.read_raw_edf(filename, verbose=False)
        dataAsNdArray = rawData.get_data()
        return dataAsNdArray
    except Exception as e:
        print('Error reading edf file: {}'.format(e))


def preprocessEdf(rawEdf: edf.RawEDF) -> pd.DataFrame:
    print('preprocessing...')
    # TODO: return preprocessed edf (as a dataframe)...
    dataframe = pd.DataFrame()
    return dataframe
