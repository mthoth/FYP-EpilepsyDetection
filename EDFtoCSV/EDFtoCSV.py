import pyedflib
import pandas as pd
import helpers as hp


""" Please make sure to have a folder outside the repo as follows: 
    'Dataset/chbxx/' (xx = patient no.) 
    and 'Dataset/Dataset_Excelsheets/'.
    ---
    Make sure YOU CREATE THE FOLDERS before running this module.
"""


def convertPatientFiles(prefix, patientDir):
    fullPath = f'{prefix}/{patientDir}'
    destinationPath = f'{prefix}/Dataset_Excelsheets/{patientDir}'

    filesInDir, dirLength = hp.getFilesInDirectory(fullPath)
    print(f'Entering directory: {fullPath} ---- dirLength: {dirLength} \n')

    for currFile in filesInDir:
        if not currFile.endswith('.edf'):
            continue

        filePath = r'{}/{}'.format(fullPath, currFile)
        print(f'Reading {currFile}...', end="\n\t")

        rawEdf = pyedflib.EdfReader(filePath)
        rawDf = pd.DataFrame()
        signals = rawEdf.getSignalLabels()
        uniqueSignals = set(signals)

        hp.appendSignalsToDataframe(uniqueSignals, rawDf, rawEdf)
        print(f'Converted edf to dataframe shape:{rawDf.shape}...', end="\n\t")

        csvName = f'{currFile[:8]}.csv'
        hp.convertDataframeToCsv(rawDf, csvName, uniqueSignals)
        hp.moveFileToDestination(src=csvName, dest=destinationPath)
    print(f'patient {patientDir} complete')


def main():
    pathPrefix = r'../Dataset'
    # These should be changed to YOUR specific patients
    # patientDirs = (r'chb18', r'chb19', r'chb20', r'chb21', r'chb22')
    patientDirs = ('chb24',)

    for currPatientDir in patientDirs:
        convertPatientFiles(pathPrefix, currPatientDir)

    print('FILE CONVERSIONS COMPLETE!')


if __name__ == "__main__":
    main()
