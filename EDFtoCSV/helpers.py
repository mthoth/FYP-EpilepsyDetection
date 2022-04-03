import os
import shutil


def getFilesInDirectory(path):
    listOfFiles = os.listdir(path)
    return (listOfFiles, len(listOfFiles))


def convertDataframeToCsv(dataframe, targetName, headerNames):
    print(f'Converting to final csv...', end="\n\t")
    dataframe.to_csv(targetName, header=headerNames)
    print('conversion done.')


def appendSignalsToDataframe(signals, dataframe, edf):
    def isAValidSignalName(sigName):
        return sigName[0].isalpha()

    i = 0
    for currSignal in signals:
        if isAValidSignalName(currSignal) and i < 18:
            dataframe.insert(i, currSignal, edf.readSignal(i))
            i += 1


def moveFileToDestination(src, dest):
    destinationDir = r'{}/{}'.format(dest, src)
    try:
        if not os.path.exists(destinationDir):
            os.makedirs(destinationDir)

        shutil.move(src, destinationDir)
        print(f'successfully saved to {destinationDir}.\n')
    except:
        print('Moving file failed.')
