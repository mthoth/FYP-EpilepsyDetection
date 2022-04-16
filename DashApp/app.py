import numpy as np
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate, CallbackException
import dash_bootstrap_components as dbc

import helpers as hp
from src.myComponents import Container
from src import graphData

app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

# App layout, where we put the components
appStyles = {'textAlign': 'center', 'maxWidth': '80rem', 'margin': 'auto'}
app.layout = html.Div([
    Container([
        html.H1(f"Seizure Detection Test"),
        html.Br(),

        html.Div([
            html.H5("Click below to start example on Patient chb20"),
            btnStartModel := dbc.Button("Start", size="lg", color="success", n_clicks=0)
        ]),

        modelStatus := html.H5(),

        html.Div([
            seizureFigure := dcc.Graph(),
            graphInterval := dcc.Interval(interval=2*1000, n_intervals=0),
            patientAlert := dbc.Alert(graphTimer := html.H5(), color="secondary")
        ])
    ]),
], style=appStyles)


@app.callback(
    Output(seizureFigure, "figure"),
    Output(graphTimer, "children"),
    Output(graphInterval, "disabled"),
    Output(btnStartModel, "disabled"),
    Output(modelStatus, "children"),
    Output(patientAlert, "color"),
    Input(graphInterval, "n_intervals"),
    Input(btnStartModel, "n_clicks")
)
def updateGraphOutput(currentInterval, nClicksInput):
    def getInitCallbackOutputs(clicks):
        return (False, True, "Running model...") if clicks > 0 else (True, False, None)

    def getAlertColor(figureY):
        return "danger" if graphData.getPatientStatus(figureY) == "DANGER" else "secondary"

    graphIntervalIsDisabled, btnStartModelIsDisabled, modelStatus = getInitCallbackOutputs(
        nClicksInput)

    # if btnStartModelIsDisabled:
    #     figX, figY = runModel()
    fig, figX, figY = graphData.getUpdatedGraph(currentInterval)
    alertColor = getAlertColor(figY)

    try:
        currentGraphTime = figX[-1]
        graphStatus = f'Elapsed Time: {currentGraphTime}s'
    except Exception:
        graphStatus = "Data stream ended. To run again, refresh the page."
        alertColor = "warning"
        modelStatus = None

    return fig, graphStatus, graphIntervalIsDisabled, btnStartModelIsDisabled, modelStatus, alertColor


# Entry point of program
if __name__ == '__main__':
    app.run_server(debug=True)
