import datetime
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor

#
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)


# # create function to use XGBRegressor for next day prediction
# def xgbr_predict(data_df, train_start_date, window_size=6):
#     """This module will use the Closing priice of a cryptocurrency 
#     to predict the next day price using XGBRegressor.

#     INPUTS: 
#     	- data_df: Historical data from yfinance
#     	- train_strat_dat: date used to split data into training and testing sets
#     	- window_size: how many past days to consider for a one-step prediction

#     OUTPUT: 
#     	- plot_df: a data frame with the actual closing prices used in the testing 
#     	and predicted closing prices
#     	- pred[-1] the predicted value for next day
#     """
#     df = data_df[["Close"]].copy()
#     n = train_start_date

#     for i in range(1, window_size+1):
#         df[f'Close-{i}'] = df['Close'].shift(i)

    
#     df = df.dropna()
#     # features and target 
#     y = df[["Close"]]
#     X = df.drop(columns=["Close"])
#     # train test split
#     X_train = X.loc[:n]
#     y_train = y.loc[:n]
#     X_test = X.loc[n:]
#     y_test = y.loc[n:]

#     # train and fit model
#     model = XGBRegressor(
#         objective="reg:squarederror", 
#         n_estimators=1000, 
#         learning_rate=0.01)

#     model.fit(X_train, y_train)
#     # predict
#     pred = model.predict(X_test)
#     # make plot df
#     plot_df = pd.DataFrame(y_test)
#     plot_df["Pred"] = pred
#     plot_df.index = y_test.index

#     # test0 = test.loc["2022-08-08"]

#     # return plot_df and the last predicted value, pred[-1]
#     return plot_df, pred[-1]


# ================
# define functions
# ================
# create features
def create_features(data, window_size):
    '''
    data is the Close price from the historical data from yfinance 
    with the date as index
    '''
    df = data.copy()
    for i in range(1, window_size+1):
        df[f"Close-{i}"] = df["Close"].shift(i)
    
    df.dropna(inplace=True)
    return df

# split features table into training and test,
# and return features and target for training and test
def tts(df, train_end_date):
    '''
    df is the features table created from the Close price column
    with a given window size
    '''

    # first split data into training and test sets
    n = train_end_date
    # n = test_start_date
    train = df.loc[:n]
    test = df.loc[n:]

    # training features and target
    y_train = train["Close"].values
    X_train = train.drop(columns="Close").values
    # test features and target
    y_test = test["Close"].values
    X_test = test.drop(columns="Close").values

    return X_train, y_train, X_test, y_test, test


# ===========================
# define prediction function
# ===========================
def xgbr_predict(data, train_end_date, window_size=6):

    # create features
    df = create_features(data, window_size)

    # train/test split
    X_train, y_train, X_test, y_test, test_df = tts(df, train_end_date)

    # train model 
    model = XGBRegressor(
            objective="reg:squarederror", 
            n_estimators=1000, 
            learning_rate=0.01)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # create test/prediction plot 
    plot_df = test_df.copy()
    plot_df["Prediction"] = y_pred
    # fig_test_pred = px.line(plot_df, y=["Close", "Prediction"])

    # create features for nextday prediction
    features = test_df.loc["2022-08-20"].values

    # X_today = features[:-1].reshape(1, window_size)
    # predction for tomorrow
    # y_nextday = model.predict(X_today)

    # return test/pred plot and neextdat prediction
    return plot_df, y_pred[-1]

