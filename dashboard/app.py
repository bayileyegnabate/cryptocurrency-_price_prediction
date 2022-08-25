import datetime
import json
import numpy as np
import pandas as pd
import dash
from dash import Input, Output, dcc, html, ctx
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
import pandas_datareader.data as web
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgb_model import xgb_train_model, xgb_prediction_on_test
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# CONSTANTS
WINDOW_SIZE = 2
# Get data
start = datetime.datetime(2014,9,15)
end = datetime.date.today()
tomorrow = end + datetime.timedelta(days=1)
df = web.DataReader(
    ["BCH-USD","BNB-USD", "ETH-USD", "ADA-USD", "XRP-USD", "ETC-USD", "BTC-USD", "KYL-USD", "AVAX-USD", "SOL-USD", "DOGE-USD"], 
    "yahoo",
    start=start,
    end=end
    )
df = df.stack().reset_index()
# ================================================
# Test start dates for model perfrmance avaluation
# ================================================
test_start_dates = {
    "BCH-USD": "2021-05",
    "BTC-USD": "2021-06",
    "ETH-USD": "2021-06",
    "BNB-USD": "2021-12",
    "SOL-USD": "2021-12",
    "AVAX-USD": "2021-12",
    "DOGE-USD": "2021-09",
    "ADA-USD": "2021-09",
    "XRP-USD": "2021-09",
    "ETC-USD": "2021-09",
    "BTC-USD": "2021-09",
    "KYL-USD": "2021-11",
}

#===========
# start Dash
#===========
app = dash.Dash(__name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    meta_tags=[{'name':'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
    title="cryptocurrency-price-prediction-machine-learning")
server = app.server

# ==========
# app layout
# ==========
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            html.H1("Cryptocurrency Price Predictor", className="app-title py-3"),
            html.P("Get ahead of the industry by predicting the Next Day Close Price for your most bullish cryptocurrencies", className="lead-text"),
            html.Hr(),
            html.Br()
            ], className="main-header mb-4 py-3")]),
    dbc.Row([
        # select coin
        dbc.Col([
            dbc.Row([
                dbc.InputGroup([
                    html.H5("Select Cryptocurrency: ", className="text-primary mb-2"),
                    dcc.Dropdown(id="select-coin", multi=False, value="BCH-USD",
                        options=[{'label': x, 'value':x} for x in sorted(df["Symbols"].unique())],className="dropdown")], className="inputgroup mb-5")
                ]),
            dbc.Row([
                html.Div([
                    html.H4("Model Performance")
                    ], className="model-performance mb-3 mt-5")
                ]),
            dcc.Graph(id="pred-fig", figure={}),
            dbc.Button("Tomorrow's Prediction", 
                id="btn-nclicks-1", 
                n_clicks=0, outline=True,
                className="run-btn me-1 my-4"),
            html.Div(id="nextday-price-prediction", className="nextday-price mt-1"),
            ], className="first-col"),
        # time series and weekly volatility plots
        dbc.Col([
            dcc.Graph(id="tseries-fig", figure={}, className="mb-3"),
            dcc.Graph(id="volatility-fig", figure={}),
            ])
        ]),
    ],className="page-container", fluid=False)


# =========
# callbacks
# =========
@app.callback(
    Output("tseries-fig", "figure"),
    Output("nextday-price-prediction", "children"),
    Output("pred-fig", "figure"),
    Output("volatility-fig", "figure"),
    Input("btn-nclicks-1", "n_clicks"),
    Input("select-coin", "value")
    )
def update_graphs_prediction(runpred, coin_selected):
    dff = df[df["Symbols"] == coin_selected]
    # simple moving average using pandas rolling window
    dff["SMA_6Month"] = dff["Close"].rolling(window=180).mean()
    # cumulative moving average
    dff["CMA"] = dff["Close"].expanding(min_periods=3).mean()
    ts_fig = px.line(dff, x="Date", y=["Close", "SMA_6Month", "CMA"], title="Time Series", labels={"value": "Price (USD)"})
    # weekly volatility
    dff["Weekly_Voltaility"] = dff["Close"].pct_change().rolling(7).std()
    wv_fig = px.line(dff, x="Date", y="Weekly_Voltaility", title="Weekly Voltaility", labels={"Weekly_Voltaility": "Weekly Voltaility"})

    # nextday Close price
    nextday_price = ""

    test_start_date = test_start_dates[coin_selected]
    dff.set_index("Date", inplace=True)
    dff = dff[["Close"]].copy()

    coin_saved_model = f"static/{coin_selected}.json"

    plot_df, y_nextday = xgb_prediction_on_test(dff, coin_saved_model, test_start_date, WINDOW_SIZE)
    pred_test_fig = px.line(plot_df, y=["Close", "Prediction"], title="Actual vs. Predicted Close Prices", labels={"value": "Close Price (USD)", "Close": "Actual"})

    # update layouts
    for fig in [ts_fig, wv_fig, pred_test_fig]:
        fig.update_layout(
            font=dict(
                family="Open Sans",
                size=14),
            title=dict(
                x=0.5,
                y=1,
                xanchor='center',
                yanchor= 'top',
                font=dict(
                    color="orange",
                    size=24)),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.9,
                xanchor="left",
                x=0.2),
            legend_title_text='',
            margin=dict(l=20, r=20, t=30, b=30),
            # plot_bgcolor = '#DCDCDC',
            plot_bgcolor = '#eee',
            paper_bgcolor="#fff")
        fig.update_xaxes(tickangle=-45)
        

    # if tomorrow's prediction button is clicked:
    if "btn-nclicks-1" == ctx.triggered_id:
        nextday_price = f"${y_nextday:.2f}"

    # returns
    return ts_fig, html.Div(nextday_price), pred_test_fig, wv_fig


#=====
# main
# ====
if __name__ == '__main__':
    app.run_server(debug=True)