import mne
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
# pd.set_option("display.max_rows", None, "display.max_columns", None)
edf = mne.io.read_raw_edf('chb01_03.edf')
raw_edf = edf.get_data()
t = raw_edf[0:18, :]
t = t.T
before = 0
after = 512
channels = 18
time = 6
M = 8
fs = 256  # the sampling frequency of the data
h = raw_edf.shape

n = int((h[1] / 512) - 2)

num_seizures = 0  # input the number of seizures -1
seizure_count = 0
seizure_start = [2996]  # input the seizure start times in order
seizure_stop = [3036]  # input the seizure stop times in order

transformedData = np.zeros((n, 434))
bands = np.linspace(0.5, 25, num=M + 1)


def GetEpoch(before_in, after_in, index, row):
    for channel in range(18):
        x = t[before_in:after_in, channel]
        for j in range(M):
            transformedData[row, index] = bandpower(x, fs, [bands[j], bands[j+1]], 2)
            index = index + 1

    before = after_in + 1
    after = before + 511

    return before, after


# def bandpower(x, fs, fmin, fmax):
#    f, Pxx = scipy.signal.periodogram(x, fs=fs)
#    ind_min = np.argmax(f > fmin) - 1
#    ind_max = np.argmax(f > fmax) - 1
#    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

   Parameters
   ----------
   data : 1d-array
       Input signal in the time-domain.
   sf : float
       Sampling frequency of the data.
   band : list
       Lower and upper frequencies of the band of interest.
   window_sec : float
       Length of each window in seconds.
       If None, window_sec = (1 / min(band)) * 2
   relative : boolean
       If True, return the relative power (= divided by the total power of the signal).
       If False (default), return the absolute power.

   Return
   ------
   bp : float
       Absolute or relative band power.
   """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


before, after = GetEpoch(before, after, 0, 0)
before, after = GetEpoch(before, after, 144, 0)
before, after = GetEpoch(before, after, 288, 0)

if seizure_start[seizure_count] < time <= seizure_stop[seizure_count]:
    transformedData[0, 432] = 1

transformedData[0, 432] = 0
transformedData[0, 433] = time
time = time + 2

for row in range(1, n):  # n = height(T) / number of rows per epoch
    print(row)
    transformedData[row, 433] = time
    if time > seizure_stop[seizure_count] and seizure_count < num_seizures:
        seizure_count = seizure_count + 1
    transformedData[row, 0:288] = transformedData[row - 1, 144:432]
    before, after = GetEpoch(before, after, 288, row)
    if seizure_start[seizure_count] < time <= seizure_stop[seizure_count]:
        transformedData[row, 432] = 1
    else:
        transformedData[row, 432] = 0
    time = time + 2

pd.DataFrame(transformedData).to_csv('out.csv')
