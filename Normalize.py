def Normalize(file):

    from sklearn import preprocessing
    import numpy as np
    import pandas as pd

    data = pd.read_csv(file)
    shape = np.shape(data)

    data2 = pd.DataFrame.to_numpy(data)
    output = data2[1:shape[0], shape[1]-1:shape[1]]
    data2 = data2[1:shape[0], :shape[1]-1]
    normalized_array = preprocessing.normalize(data2)
    normalized_array = pd.DataFrame(normalized_array)
    normalized_array[432] = output
    pd.DataFrame(normalized_array).to_csv('normal_out1.csv', index=False)
