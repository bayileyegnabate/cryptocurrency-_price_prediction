'''
The Dash app uses pandas web data reader to get crypto prices 
from https://finance.yahoo.com/

Data source to be replaced database (AWS)
'''
import datetime
import numpy as np
import pandas as pd
import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas_datareader.data as web
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgbr_predict import predict_nextday_price

# Get data
start = datetime.datetime(2014,9,15)
end = datetime.date.today()
tomorrow = end + datetime.timedelta(days=1)
df = web.DataReader(
	["BNB-USD", "ETH-USD", "BTC-USD", "AVAX-USD", "SOL-USD", "DOGE-USD"], 
	"yahoo",
	start=start,
	end=end
	)
df = df.stack().reset_index()

# ================================================================================
# Train start dates depend on the coin for they have different histroical profiles.
# ================================================================================
train_starts = {
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
	title="cryptocurrency-prices-prediction")

# ==========
# components
# ==========
# title
page_title = html.H1("Cryptocurrency Price Prediction", className="pg-title text-center my-1")
# hr1
hr1 = html.Hr(className="hr_1 py-3, mb-4")
# dropdown
dropdown = dcc.Dropdown(id="select-coin", multi=False, value="BNB-USD",
				options=[{'label': x, 'value':x} for x in sorted(df["Symbols"].unique())])
# historical time series graph
timeseries_graph = dcc.Graph(id="tseries-fig", figure={})
#button to run next day prediction
run_button = html.Button("Run Modeling", id="run-button", className="btn btn-secondary btn-large mb-4 p-2", value="BNB-USD", disabled=True)
# display next day predicted price
next_price = html.Div([
				html.H4("Tomorow's Prediction:", className="pred-h4"),
				html.Div(id='price-output', className="price-output",)
				])
# prediction against test dataset plot
pred_graph = dcc.Graph(id="pred-fig", figure={})


# ======
# Layout
# ======
app.layout = dbc.Container([
	dbc.Row([
		dbc.Col([
			page_title,
			hr1],
			width=12)
		]),
	dbc.Row([
		dbc.Col([
			dropdown,
			timeseries_graph
			],xs=12,sm=12,md=12,lg=5,xl=5 
			),
		dbc.Col([
			run_button,
			html.Br(),
			next_price,
			pred_graph
			],xs=12,sm=12,md=12,lg=5,xl=5
			),
		],justify="center")
	])
# ======= end layout =======

# =========
# callbacks
# =========
# time series plot
@app.callback(
	Output("tseries-fig", "figure"),
	Input("select-coin", "value")
	)

def update_graph(coin_selected):
	dff = df[df["Symbols"] == coin_selected]
	fig_1 = px.line(dff, x="Date", y="Close", title="Historical Prices")

	return fig_1

# price prediction
@app.callback(
	Output("pred-fig", "figure"),
	Output(component_id='price-output', component_property='children'),
	Input("select-coin", "value")
	)

def update_graph(coin_selected):
    df1 = df[df["Symbols"] == coin_selected]
    df1.set_index("Date", inplace=True)
    df1 = df1[["Close"]].copy()
    train_date = train_starts[coin_selected]
    plot_df, pred_price = predict_nextday_price(df1, train_date, 6)
    fig_pred = px.line(plot_df, y=["Close", "Pred"], title="Actual vs Predicted",
    	width=700, height=400,
    	labels={"value": "Coin Value (USD)", "Date": ""}
    	)

    fig_pred.update_layout(
    font=dict(
        family="Open Sans",
        size=13,
        color="RebeccaPurple"
    ),
    title=dict(
        text="Actual vs Predicted",
        x=0.5,
        y=0.9,
        xanchor='center',
        yanchor= 'top',
        font=dict(
            family="Open Sans",
            color="grey",
            size=22,
        )
    ),
    legend_title="Prices",
    legend_title_font_color="green"
    )
    fig_pred.update_layout(
    	margin=dict(l=18, r=18, t=18, b=18),
    	)
    fig_pred.update_xaxes(tickangle=-45)

    return fig_pred, f"${pred_price:.2f}"
# ======= end layout ====================

# main
if __name__ == '__main__':
	app.run_server(debug=True, port=3000)