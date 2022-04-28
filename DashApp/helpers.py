import io
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
        preprocessedDf = readEdfAndConvertToDataframe(io.BytesIO(decoded))
        # TODO: The conversion to df worked on the above example. Take the csv and do a prediction on it based on a selected model from the dropdown
        # io.BytesIO(decoded.decode('utf-8'))
        # io.StringIO(decoded.decode('utf-8'))
    except Exception as e:
        print(e)
        return ErrorText("There was an error processing this file")

    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),

        html.Hr(),

        html.Div('Raw Content, for Debugging'),
        html.Pre(contents[0:300] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
