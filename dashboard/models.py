import datetime
import pandas as pd
from xgboost import XGBRegressor

#
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)


# create function to use XGBRegressor for next day prediction
def xgbr_predict(data_df, train_start_date, window_size=6):
    """This module will use the Closing priice of a cryptocurrency 
    to predict the next day price using XGBRegressor.

    INPUTS: 
    	- data_df: Historical data from yfinance
    	- train_strat_dat: date used to split data into training and testing sets
    	- window_size: how many past days to consider for a one-step prediction

    OUTPUT: 
    	- plot_df: a data frame with the actual closing prices used in the testing 
    	and predicted closing prices
    	- pred[-1] the predicted value for next day
    """
    df = data_df[["Close"]].copy()
    n = train_start_date

    for i in range(1, window_size+1):
        df[f'Close-{i}'] = df['Close'].shift(i)

    
    df = df.dropna()
    # features and target 
    y = df[["Close"]]
    X = df.drop(columns=["Close"])
    # train test split
    X_train = X.loc[:n]
    y_train = y.loc[:n]
    X_test = X.loc[n:]
    y_test = y.loc[n:]

    # train and fit model
    model = XGBRegressor(
        objective="reg:squarederror", 
        n_estimators=1000, 
        learning_rate=0.01)

    model.fit(X_train, y_train)
    # predict
    pred = model.predict(X_test)
    # make plot df
    plot_df = pd.DataFrame(y_test)
    plot_df["Pred"] = pred
    plot_df.index = y_test.index

    # test0 = test.loc["2022-08-08"]

    # return plot_df and the last predicted value, pred[-1]
    return plot_df, pred[-1]

