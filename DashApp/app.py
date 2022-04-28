import numpy as np
import pandas as pd
from mne.io.edf.edf import RawEDF
import plotly.express as px
from dash import Dash, html, dcc, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from src.myComponents import Container
from src import graphData
from src.graphData import getUpdatedGraph, initEegData

app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

raw = initEegData()

historyContentGlobal = list()

# App layout
appStyles = {'textAlign': 'center', 'maxWidth': '92rem', 'margin': 'auto'}
app.layout = html.Div([
    Container([
        html.H1(f"Seizure Detection Test"),

        dbc.Row([
            dbc.Col([
                patientAlert := dbc.Alert(graphTimer := html.H5(), color="secondary"),
                html.Div([
                    seizureGraph := dcc.Graph(),
                    eegGraph := dcc.Graph(),
                    graphInterval := dcc.Interval(interval=2*1000, n_intervals=0),
                ]),
            ], width=9),

            dbc.Col([
                html.H4(f"Seizure History"),
                historyTable := dbc.Table([
                    html.Thead([html.Th('Time of Seizure'), html.Th('True Positive?')]),
                    historyTableContent := html.Tbody(historyContentGlobal)
                ], bordered=True, hover=True, striped=True)
            ], width=3, style={"max-height": "100vh", "overflow-y": "auto"})
        ]),
    ]),
], style=appStyles)



@app.callback(
    Output(eegGraph, "figure"),
    Output(graphTimer, "children"),
    Input(graphInterval, "n_intervals"),
)
def updateEegOutput(currentInterval):
    def getEegFigure(rawEdf: RawEDF, currTime):
        secondStart, secondEnd = currTime*2, (currTime*2) + 8

        start, stop = rawEdf.time_as_index([secondStart, secondEnd])
        data, times = rawEdf[rawEdf.ch_names[:], start:stop]

        return px.line(pd.DataFrame({
            "Time (s)": times,
            "EEG data (V)": data.T[:, 0],
        }),
            x="Time (s)", y="EEG data (V)")

    _fig, figX, _figY = getUpdatedGraph(currentInterval)
    eegFig = getEegFigure(raw, currentInterval)

    try:
        currentGraphTime = figX[-1]
        graphStatus = f'Elapsed Time: {currentGraphTime}s'
    except Exception:
        _fig = eegFig = no_update
        graphStatus = "Data stream ended. To run again, refresh the page."

    return eegFig, graphStatus


@app.callback(
    Output(seizureGraph, "figure"),
    Output(patientAlert, "color"),
    Input(graphInterval, "n_intervals")
)
def updateSeizureGraph(n):
    def getAlertColor(figureY):
        return "danger" if figureY[-1] == 1 else "secondary"


    def getPatientStatus(figureY):
        try:
            return "DANGER" if figureY[-1] == 1 else "OK"
        except Exception:
            # ? When figureY[-1] is out of bounds
            return "???"

    try:
        graphCoords = graphData.getPredictions2(n)
        x_time, y_label = graphCoords["x"], graphCoords["y"]
        alertColor = getAlertColor(y_label)
        
        fig = px.line(pd.DataFrame({
            'Time (s)': x_time,
            'Seizure Label': y_label
        }),
            x="Time (s)", y='Seizure Label', title=f'Patient Status: {getPatientStatus(y_label)}', line_shape='hv', range_y=["-0.1", "1.1"])
        return fig, alertColor

    except Exception as e:
        alertColor = "warning"
        return no_update, alertColor



@app.callback(
    Output(historyTableContent, "children"),
    Input(graphInterval, "n_intervals"),
)
def updateHistory(n):
    global historyContentGlobal
    try:
        graphCoords = graphData.getPredictions2(n)
        x_time, y_label = graphCoords["x"], graphCoords["y"]
        if y_label[-1] == 1:
            CORRECT, WRONG = (("YES", "#e8ffe8"), ("NO", "#ffe8e8"))
            labelVal, cellColor = CORRECT if graphData.getActualY(x_time[-1]) == y_label[-1] else WRONG

            historyContentGlobal.append(html.Tr([html.Td(f"{x_time[-1]}s"), html.Td(labelVal)], style={"background-color": cellColor}))
        return historyContentGlobal
    except Exception as e:
        print(f'[ERROR @updateHistory]: {e}')
        raise PreventUpdate



# Entry point of program
if __name__ == '__main__':
    app.run_server(debug=True)
