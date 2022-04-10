from dash import html, dcc


def Container(children, padding="1rem 5rem"):
    return html.Div(children=children, style={'padding': '1rem 5rem', 'textAlign': 'center'})


def FileUploader(id):
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
        style=uploaderStyle)


def GraphContainer(figure):
    return html.Div([
        dcc.Graph(figure=figure)
    ])


def ErrorText(children):
    return html.H5(children=children, className='text-error')
