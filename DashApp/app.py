from datetime import datetime
import io
import base64
import pandas as pd

import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from helpers import parseContents

from src.edfProcessing import *
from src.myComponents import Container, FileUploader
from src.graphData import fetchLiveData

stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=stylesheets)

# App layout, where we put the components
appStyles = {'textAlign': 'center', 'maxWidth': '80rem', 'margin': 'auto'}
app.layout = html.Div([
    Container([
        html.H1(f'Seizure Detection Test'),
        html.P('Dash: A web application framework your data.'),
        FileUploader(id="data-upload"),
        html.Div(id='output-data-upload'),
        html.Div([
            dcc.Graph(id="my-graph"),
            dcc.Interval(id="my-interval", interval=2*1000, n_intervals=0),
            html.H3(id="graph-time", children=None)
        ])
    ]),
], style=appStyles)


@app.callback(
    Output('output-data-upload', 'children'),
    Input('data-upload', 'contents'),
    State('data-upload', 'filename'),
    State('data-upload', 'last_modified')
)
def updateCsvOutput(listOfContents, listOfNames, listOfDates):
    if listOfContents is not None:
        children = [
            parseContents(c, n, d) for c, n, d in
            list(zip([listOfContents], [listOfNames], [listOfDates]))]
        return children


@app.callback(
    Output('my-graph', 'figure'),
    Output('graph-time', 'children'),
    Input('my-interval', 'n_intervals')
)
def updateGraphOutput(n):
    def getPatientStatus(figureY):
        return "DANGER" if figureY[-1] == 1 else "OK"

    idxEnd = n + 10
    figX, figY = fetchLiveData(n)

    # if idxEnd > len(figX):
    #     raise PreventUpdate

    fig = px.line(pd.DataFrame({
        'Time (s)': figX,
        'Seizure Label': figY
    }),
        x="Time (s)", y='Seizure Label', title=f'Patient Status: {getPatientStatus(figY)}', line_shape='hv', range_y=["-0.1", "1.1"])

    currentGraphTime = figX[-1]
    return fig, f'Elapsed Time: {currentGraphTime}s'


# Entry point of program
if __name__ == '__main__':
    app.run_server(debug=True)
