import os
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate, CallbackException
import dash_bootstrap_components as dbc

import helpers as hp
from src.myComponents import Container, FileUploader
from src.graphData import fetchLiveData
import src.training2 as trn

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
            patientAlert := dbc.Alert(graphTimer := html.H4(id="graph-time"), color="secondary")
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

    def getPatientStatus(figureY):
        try:
            return "DANGER" if figureY[-1] == 1 else "OK"
        except Exception:
            # ? When figureY[-1] is out of bounds
            return "???"

    def getAlertColor(figureY):
        return "danger" if getPatientStatus(figureY) == "DANGER" else "secondary"

    def getUpdatedGraph(currTime):
        figX, figY = fetchLiveData(currTime)
        fig = px.line(pd.DataFrame({
            'Time (s)': figX,
            'Seizure Label': figY
        }),
            x="Time (s)", y='Seizure Label', title=f'Patient Status: {getPatientStatus(figY)}', line_shape='hv', range_y=["-0.1", "1.1"])

        return fig, figX, figY

    def runModel():
        try:
            # ! To be changed depending on where you have the files on your machine
            PREPROCESSED_DIRECTORY = f'DashApp/data/preprocessed'
            preprocessedFiles = [file for file in os.listdir(
                PREPROCESSED_DIRECTORY) if file.endswith('.csv')]

            Xtrain, ytrain, Xtest, _ = trn.noisyData(
                PREPROCESSED_DIRECTORY, preprocessedFiles)
            fitModel = trn.SVM_CLASSWEIGHT_BALANCED(Xtrain, ytrain)
            yPred: np.ndarray = fitModel.predict(Xtest)

            # ! THIS SHOULD BE WHAT I WANT FOR GETTING VALUES NEAR THE FIRST SEIZURE OCCURENCE
            timingsLabels = np.array([(2*i + 6, label)
                                     for i, label in enumerate(yPred)])
            timings, labels = timingsLabels[:, 0], timingsLabels[:, 1]
            seizuresIndices = np.where(labels == 1)
            idxOfFirstSeizure = seizuresIndices[0][0]
            # idxSeizuresPred = [idx for idx,
            #                    label in enumerate(yPred) if label == 1]
            # # yTimings = [i*2 + 6 for i, label in enumerate(yPred)]

            # FOR TESTING
            return timings[idxOfFirstSeizure-10:idxOfFirstSeizure+10], labels[idxOfFirstSeizure-10:idxOfFirstSeizure+10]
            # BASE CASE
            # return timings, labels
        except Exception:
            print('Failed to train and test')

    graphIntervalIsDisabled, btnStartModelIsDisabled, modelStatus = getInitCallbackOutputs(
        nClicksInput)

    # if btnStartModelIsDisabled:
    #     figX, figY = runModel()
    fig, figX, figY = getUpdatedGraph(currentInterval)
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
