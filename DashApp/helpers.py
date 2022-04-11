# import io
import base64
from datetime import datetime
from dash import html

from src.edfProcessing import readEdfAndConvertToDataframe
from src.myComponents import ErrorText


def parseContents(contents, filename, date):
    if not filename.endswith('.edf'):
        return ErrorText("File format MUST be 'edf'")

    contentType, contentString = contents.split(',')
    decoded = base64.b64decode(contentString)

    try:
        preprocessedDf = readEdfAndConvertToDataframe(
            '../Dataset/chb20/chb20_14.edf')
        print('successful read of EDF file:')
        # TODO: The conversion to df worked on the above example. Take the csv and do a prediction on it based on a selected model from the dropdown
        # io.StringIO(decoded.decode('utf-8'))

    except Exception as e:
        print(e)
        return ErrorText("There was an error processing this file")

    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
