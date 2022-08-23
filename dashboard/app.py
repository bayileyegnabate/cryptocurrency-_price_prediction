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
# from xgb_model import xgb_price_predictor, xgb_prediction_on_test
from xgb_model import xgb_prediction_on_test

# CONSTANTS
WINDOW_SIZE = 2

# Get data
start = datetime.datetime(2014,9,15)
end = datetime.date.today()
tomorrow = end + datetime.timedelta(days=1)
df = web.DataReader(
    ["BCH-USD","BNB-USD", "ETH-USD", "BTC-USD", "AVAX-USD", "SOL-USD", "DOGE-USD"], 
    "yahoo",
    start=start,
    end=end
    )
df = df.stack().reset_index()
# ================================================================================
# Train start dates depend on the coin for they have different histroical profiles.
# ================================================================================
train_end_dates = {
    "BCH-USD": "2021-05",
    "BTC-USD": "2021-06",
    "ETH-USD": "2021-06",
    "BNB-USD": "2021-12",
    "SOL-USD": "2021-12",
    "AVAX-USD": "2021-12",
    "DOGE-USD": "2021-09",
}

#===========
# start Dash
#===========
app = dash.Dash(__name__, 
    # external_stylesheets=[dbc.themes.BOOTSTRAP], 
    external_stylesheets=[dbc.themes.MATERIA], 
    meta_tags=[{'name':'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
    title="local-cryptocurrency-price-prediction-machine-learning")
server = app.server

# layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            html.H1("Cryptocurrency Price Prediction", className="app-title pb-3"),
            html.P("Using Machine Learning to Predict Next Day Price", className="lead_text")
            ], className="main-header mb-4 py-3 text-center")]),
    dbc.Row([
        # select coin
        dbc.Col([
            dcc.Dropdown(id="select-coin", multi=False, value="BCH-USD",
                options=[{'label': x, 'value':x} for x in sorted(df["Symbols"].unique())]),
            dbc.Button("Run Prediction", 
                id="btn-nclicks-1", 
                n_clicks=0, outline=True,
                className="run-btn me-1 mt-5"),
            html.Div(id="nextday-price-prediction", className="nextday-price mt-5"),
            dcc.Graph(id="pred-fig", figure={}),
            # model info
            dbc.Row([
                dbc.Col([
                    html.Div([
                    html.H5("96.8%"),
                    html.P("Model Training Accuracy")],className="model-info-item py-2")
                    ]),
                dbc.Col([
                    html.Div([
                    html.H5("91.3%"),
                    html.P("Model Test Accuracy")],className="model-info-item py-2")
                    ])
                ], className="model-info mt-5")
            ]),
        # time series and weekly volatility plots
        dbc.Col([
            dcc.Graph(id="tseries-fig", figure={}),
            dcc.Graph(id="volatility-fig", figure={}),
            ])
        ]),
    ],className="page-container", fluid=False)


# =========
# callbacks
# =========

# time series
@app.callback(
    Output("tseries-fig", "figure"),
    Output("nextday-price-prediction", "children"),
    Output("pred-fig", "figure"),
    Input("btn-nclicks-1", "n_clicks"),
    Input("select-coin", "value")
    )
def update_timeseries_graph(runpred, coin_selected):
    dff = df[df["Symbols"] == coin_selected]
    # simple moving average using pandas rolling window
    dff["SMA_6Month"] = dff["Close"].rolling(window=180).mean()
    # cumulative moving average
    dff["CMA"] = dff["Close"].expanding(min_periods=3).mean()
    ts_fig = px.line(dff, x="Date", y=["Close", "SMA_6Month", "CMA"], labels={"Date": "", "Value": "Price (USD)"})

    # df1 = df[df["Symbols"] == coin_selected]
    
    # initialize nextday Close price
    nextday_price = ""

    train_end_date = train_end_dates[coin_selected]
    dff.set_index("Date", inplace=True)
    dff = dff[["Close"]].copy()

    coin_saved_model = f"static/{coin_selected}.json"

    plot_df, y_nextday = xgb_prediction_on_test(dff, coin_saved_model, train_end_date, WINDOW_SIZE)
    pred_test_fig = px.line(plot_df, y=["Close", "Prediction"], title="Actual vs Prediction")

    # if run is clicked display the nextday prediction value
    if "btn-nclicks-1" == ctx.triggered_id:
        nextday_price = f"${y_nextday:.2f}"


    return ts_fig, html.Div(nextday_price), pred_test_fig 

# weekly volatility
@app.callback(
    Output("volatility-fig", "figure"),
    Input("select-coin", "value")
    )
def update_volatility_graph(coin_selected):
    dff = df[df["Symbols"] == coin_selected]
    dff["Weekly_Voltaility"] = dff["Close"].pct_change().rolling(7).std()
    wv_fig = px.area(dff, x="Date", y="Weekly_Voltaility", title="Weekly Voltaility", labels={"Date": "", "Value": "Price (USD)"})

    return wv_fig

# # run prediction
# @app.callback(
#     Output("nextday-price-prediction", "children"),
#     # Output("pred-fig", "figure"),
#     Input("btn-nclicks-1", "n_clicks"),
#     Input("select-coin", "value")
#     )
# def update_nextday_price(runpred, coin_selected):
#     y_nextday = ""
#     pred_fig = {}
#     if "btn-nclicks-1" == ctx.triggered_id:
#         df1 = df[df["Symbols"] == coin_selected]
#         df1.set_index("Date", inplace=True)
#         df1 = df1[["Close"]].copy()
#         train_end_date = train_end_dates[coin_selected]
#         plot_df, y_nextday, coin_model = xgb_price_predictor(df1, coin_selected, train_end_date, WINDOW_SIZE)
#         pred_fig = px.line(plot_df, y=["Close", "Prediction"], title="Actual vs Prediction")
#     elif coin_selected != "value":
#         y_nextday = ""

#     return html.Div(y_nextday), pred_fig


# ******************************************************
# # predict nextday
# @app.callback(
#      Output("nextday-price-prediction", "children"),
#      Input("btn-nclicks-1", "n_clicks"),
#     )
# def update_price(runpred, coin_selected):
#     df1 = df[df["Symbols"] == coin_selected]
#     df1.set_index("Date", inplace=True)
#     df1 = df1[["Close"]].copy()
#     test_start_date = train_start_dates[coin_selected]
#     y_nextday_prediction = xgb_price_predictor(df1, coin_selected, test_start_date, WINDOW_SIZE)

#     return y_nextday_prediction

# ******************************************************



#=====
# main
# ====
if __name__ == '__main__':
    app.run_server(debug=True, port=3342)