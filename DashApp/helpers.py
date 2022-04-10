import base64
from datetime import datetime
from dash import html

from src.edfProcessing import readEdfAndConvertToDataframe
from src.myComponents import ErrorText


def parseContents(contents, filename, date):
    if not filename.endswith('.edf'):
        return ErrorText("File format MUST be 'edf'")

    contentType, constentString = contents.split(',')
    decoded = base64.b64decode(constentString)
    try:
        print('should be reading an edf file...')

        df = readEdfAndConvertToDataframe(filename)
        # * reading 'out.csv' should be replaced with the below code...
        # df = pd.read_csv(
        #     io.StringIO(decoded.decode('utf-8')))
        # preprocessedData = preprocessEeg(df)

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
