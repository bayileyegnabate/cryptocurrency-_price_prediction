import datetime
import numpy as np
import pandas as pd
import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
import pandas_datareader.data as web
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# from .models import xgbr_predict
from xgb_model import xgb_price_predictor

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
train_starts = {
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
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    meta_tags=[{'name':'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
    title="dash-app-layout")
server = app.server

# components
# ==========
# header
header = html.Div(
    [
        html.H1("Cryptocurrency Price Prediction", className="app-title pb-3"),
        html.P("Using Machine Learning to Predict Next Day Price", className="lead_text")
    ], className="main-header mb-4 py-3 text-center"
)

# first row
div_0 = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            html.H5("96.8%"),
            html.P("Model Training Accuracy")],className="model-info-item py-2")),
        dbc.Col(html.Div([
            html.H5("83.2%"),
            html.P("Model Test Accuracy")],className="model-info-item py-2")),
        # dbc.Col(html.Div([
        #     html.H3("70/15/15"),
        #     html.P("Training/Test/Validation Split")], className="model-info-item py-2")),
    ], className="model-info my-3 py-2 text-center")
    ])



# second row
div_1 = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Dropdown(id="select-coin", multi=False, value="BCH-USD",
                options=[{'label': x, 'value':x} for x in sorted(df["Symbols"].unique())]),
            dcc.Graph(id="tseries-fig", figure={})
            ]), md=5),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H5("Predicted Price", className="pred-price"),
                    html.P(id="nextday"),
                    html.Div(id="pred-price", className="nextday-price text-center my-1 px-3 py-1")
                    ], className="nextday-col text-center"),
                # dbc.Col([
                #     html.H5("Next Week Avg.", className="pred-price"),
                #     html.Div(id="nextweek-avg", className="nextday-price text-center my-1 px-3 py-1")
                #     ], className="nextday-col text-center"),
                # dbc.Col(html.H4("Predicted Price", className="pred-price")),
                ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="pred-plot", figure={}))
                ])
            ], md=5),
        ])
    ]) 

# weekly volatility
div_2 = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("Weekly Volatility"),
            # dcc.Checklist(
            #     id="btc-vol",
            #     options=["BTC Volatility"],
            #     value=["btc-voltaility"]
            #     ),
            dcc.Graph(id="volatility", figure={})
            ])      
        ], className="volt-div mb-4")
    ])

# layout
app.layout = dbc.Container([
    header,
    div_0,
    div_1,
    # html.Br(),
    div_2,
    ],className="page-container", fluid=True)


# =========
# callbacks
# =========
# 
# 
# 
# time series plot
@app.callback(
    Output("tseries-fig", "figure"),
    Input("select-coin", "value")
    )

def update_graph(coin_selected):
    dff = df[df["Symbols"] == coin_selected]
    # simple moving average using pandas rolling window
    dff["SMA_6Month"] = dff["Close"].rolling(window=180).mean()
    # cumulative moving average
    dff["CMA"] = dff["Close"].expanding(min_periods=3).mean()
    fig_1 = px.line(dff, x="Date", y=["Close", "SMA_6Month", "CMA"], title="Historical Prices",
         width=500, height=400,
        labels={"Date": "", "Value": "Coin Value (USD)"}
        )
    font=dict(
        family="Open Sans",
        size=13,
    )
    fig_1.update_xaxes(tickangle=-45)
    fig_1.update_layout(legend=dict(
        title="",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
    
    return fig_1


# predict nextday price
@app.callback(
    Output("pred-plot", "figure"),
    Output("pred-price", "children"),
    Output("nextday", "children"),
    # Output("nextweek-avg", "children"),
    Input("select-coin", "value")
    )
def update_nextday_price(coin_selected):
    dff = df[df["Symbols"] == coin_selected]

    nextday_price = np.random.randint(1000)
    nextweek_avg_price = np.random.randint(1000)

    df1 = df[df["Symbols"] == coin_selected]
    df1.set_index("Date", inplace=True)
    df1 = df1[["Close"]].copy()
    train_date = train_starts[coin_selected]
    # plot_df, pred_price = xgbr_predict(df1, train_date, 6)
    plot_df, pred_price = xgb_price_predictor(df1, train_date, 6)
    fig_pred = px.line(plot_df, y=["Close", "Pred"], title="Actual vs Predicted",
        width=600, height=400,
        labels={"Date": "", "value": "Coin Value (USD)"}
        )

    fig_pred.update_layout(
    font=dict(
        family="Open Sans",
        size=13,
        color="RebeccaPurple"
    ),
    title=dict(
        # text="Time Series",
        x=0.5,
        y=0.9,
        xanchor='center',
        yanchor= 'top',
        font=dict(
            family="Open Sans",
            color="orange",
            size=20,
        )
    ),
    legend_title="",
    legend_title_font_color="grey"
    )
    fig_pred.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # paper_bgcolor="LightSteelBlue",
        )
    fig_pred.update_xaxes(tickangle=-45)

    return fig_pred, f"${pred_price:.0f}", f"({tomorrow})"

# weekly volatility
@app.callback(
    Output("volatility", "figure"),
    # Output("btc-vol", "figure"),
    Input("select-coin", "value")
    )
def update_volatility_graph(coin_selected):
    btc_df = df[df["Symbols"] == "BTC-USD"]
    dff = df[df["Symbols"] == coin_selected]
    dff["weekly_volt"] = dff["Close"].pct_change().rolling(7).std()
    btc_df["weekly_volt"] = btc_df["Close"].pct_change().rolling(7).std()
    fig_3 = px.line(dff, x="Date", y="weekly_volt", title="Weekly Close Price Volatility",
        labels={"Date": "", "weekly_volt": "Weekly Volatility"}
        )
    # update
    fig_3.update_layout(
    font=dict(
        family="Open Sans",
        size=13,
    ),
    title=dict(
        # # text="Time Series",
        # x=0.3,
        # y=0.9,
        # xanchor='center',
        # yanchor= 'top',
        font=dict(
            family="Open Sans",
            color="orange",
            size=20,
            )
        )
    )
    fig_3.update_xaxes(tickangle=-45)

    return fig_3
# main
if __name__ == '__main__':
    app.run_server(debug=True, port=3342)
