from dash import html, dcc


def Container(children, padding="1rem 5rem"):
    return html.Div(children, style={'padding': padding, 'textAlign': 'center'})


def FileUploader(id, disabled=False):
    uploaderStyle = {
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '2rem auto',
        'maxWidth': '45rem'
    }
    return dcc.Upload(id=id, children=html.Div(
        [
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        disabled=disabled,
        style=uploaderStyle)


def GraphContainer(figure):
    return html.Div(
        dcc.Graph(figure=figure)
    )


def PatientDropdown(style):
    # TODO: choosing the patient should switch the patient
    # ! This feature might be unnecessary...
    dropdownOptions = {
        'chb20': 'Patient 20',
        'chb24': 'Patient 24',
    }

    return html.Div([
        html.Label('Select Patient', htmlFor="patient-dropdown"),
        dcc.Dropdown(dropdownOptions, value=None, id="patient-dropdown")
    ], style=style)


def ErrorText(children):
    return html.H5(children=children, className='text-error')
