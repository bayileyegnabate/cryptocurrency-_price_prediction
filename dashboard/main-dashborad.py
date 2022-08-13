import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pandas_datareader.data as web
import datetime
# 
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#
start = datetime.datetime(2014,9,17)
end = datetime.datetime(2022,8,11)
df = web.DataReader(
	["BNB-USD", "ETH-USD", "BTC-USD", "AVAX-USD"], 
	"yahoo",
	start=start,
	end=end
	)
# stack
df = df.stack().reset_index()
print(df.head())

app = dash.Dash(__name__, 
	external_stylesheets=[dbc.themes.BOOTSTRAP], 
	meta_tags=[{'name':'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
	title="cryptocurrency-prices")

app.layout = dbc.Container([
	# 1st row
	dbc.Row([
		dbc.Col(html.H1("Cryptocurrency Price Prediction",
			className="text-center text-primary my-3"),
			width=12)
		]),
# 2nd row
	dbc.Row([
		# first column
		dbc.Col([
			dcc.Dropdown(id="my-dropdown", multi=False, value="BNB-USD",
				options=[{'label': x, 'value':x} for x in sorted(df["Symbols"].unique())]),
			dcc.Graph(id="line-fig", figure={})
			], 
			xs=12,sm=12,md=12,lg=5,xl=5 
			),
		# second column
		dbc.Col([
			dbc.Button("Run", color="secondary", id="run-button", className="me-2", n_clicks=0),
			dcc.Graph(id="line-fig-2", figure={})
			], #width={'size':5, "order": 2},
			xs=12,sm=12,md=12,lg=5,xl=5
			),
		],justify="center")
	])

# historical time series plot
@app.callback(
	Output("line-fig", "figure"),
	Input("my-dropdown", "value")
	)

def update_graph(coin_selected):
	dff = df[df["Symbols"] == coin_selected]
	figln = px.line(dff, x="Date", y="Close")

	return figln

# run prediction
@app.callback(
	Output("line-fig-2", "figure"),
	Input("run-button", "n_clicks")
	)

def update_graph(coin_selected):
	df1 = df[df["Symbols"] == coin_selected]
	df1.set_index("Date", inplace=True)
	df1 = df1[["Close"]].copy()

	window_size = 3
	for i in range(1, window_size+1): 
		df1[f'Close-{i}'] = df1['Close'].shift(i)

	# dropna
	df1 = df1.dropna()
	y = df1[["Close"]]
	X = df1.drop(columns=["Close"])
	nt = "2021-12"
	X_train = X.loc[:nt]
	y_train = y.loc[:nt]
	X_test = X.loc[nt:]
	y_test = y.loc[nt:]
	model = XGBRegressor(n_estimators=1000, objectives="reg:mean_squared_error", learning_rate=0.01)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	plot_df = pd.DataFrame(y_test)
	plot_df["Pred"] = pred
	plot_df.index = y_test.index
	figln2 = px.line(plot_df, y=["Close"])

	return figln2

#  
# # if __name__ == '__main__':
app.run_server(debug=True, port=3000)