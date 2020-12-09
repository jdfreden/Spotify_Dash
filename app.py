import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from src.handler import *

app = dash.Dash(__name__)


def get_str_dtype(df, col):
    """Return dtype of col in df"""
    dtypes = ['datetime', 'bool', 'int', 'float',
              'object', 'category']
    for d in dtypes:
        if d in str(df.dtypes.loc[col]).lower():
            return d


app.layout = html.Div([
    html.Label('Like Playlist'),
    dcc.Input(id='like-playlist', type='text'),

    html.Br(),

    html.Label('Dislike Playlist'),
    dcc.Input(id='dislike-playlist', type='text'),

    html.Br(),

    html.Label('Predict Playlist'),
    dcc.Input(id='predict-playlist', type='text'),

    html.Br(),
    html.Br(),

    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Br(),

    html.Div(id='my-output-filter'),

    html.Div(id='my-output'),
], style={'columnCount': 1})


@app.callback(Output('my-output-filter', 'children'),
              Output('my-output', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('like-playlist', 'value'),
              State('dislike-playlist', 'value'),
              State('predict-playlist', 'value'), prevent_initial_callbacks=True)
def update_output(n_clicks, input1, input2, input3):
    if n_clicks > 0:
        pred = handle(input1, input2, input3)

        data = pred.to_dict('rows')
        columns = [{"name": i, "id": i, } for i in pred.columns]

        pls = [{'label': v, 'value': v} for v in pred.Playlist.unique()]

        fil = html.Div([
            html.H3("Filter Playlist"),
            dcc.Dropdown(
                options=pls,
                value = pred.Playlist.unique(),
                multi = True
            ),
            html.Br()
        ], style={'display': 'inline-block', 'width': '30%', 'margin-left': '7%'})

        dat = dt.DataTable(data=data,
                           columns=columns,
                           style_cell={"text-align": "center",
                                       "font-family": ["Helvetica", "Arial", "sans-serif"],
                                       "font-size": "90%",
                                       "padding": "5px"},
                           style_as_list_view=True,
                           sort_action="native")

        return fil, dat
    return '', ''


if __name__ == '__main__':
    app.run_server(debug=True)
