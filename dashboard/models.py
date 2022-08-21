import pandas as pd
import xgboost as xgb

# nextday price prediction function
# =================================
def xgb_price_predictor(data, training_end_date, window_size):
    # create features
    df = create_features(data, window_size)
    
    # Training / Test split
    X_train, y_train, X_test, y_test, test_date_index = tss(df, training_end_date)
    
    
    # train model
    model = xgb.XGBRegressor(objective="reg:squarederror",
                         n_estimators=400,
                         learning_rate=0.01,
                         random_state=1,
                         # early_stoping_rounds=50
                        )
    model.fit(X_train, y_train)
    
    # make prediction on the Test set
    y_pred = model.predict(X_test)
    
    # create plot_df 
    plot_df = pd.DataFrame(y_test, columns=["Close"])
    plot_df["Prediction"] = y_pred
    plot_df.index = test_date_index
    
    # make nextday prediction
    X_today = y_test[-window_size::][::-1].reshape(1, window_size)    
    y_nextday = model.predict(X_today)
    
    # return test/prediction plot and nextday prediction
    return plot_df, y_nextday[0]


# create features
# ===============
def create_features(data, window_size):
    '''
    This function converts the "Close" price column into a features table
    using a given window size
    
    INPUTS:
    ------
    data: the row data from Yahoo Finance
    window_size:
    
    OUTPUT:
    ------
    df: a data frame with the "Close"
    '''
    df = data[["Close"]].copy()
    
    for i in range(1, window_size + 1):
        df[f"Close-{i}"] = df["Close"].shift(i)
        
    df.dropna(inplace=True)
    
    return df

# training/test split
# ===================
def tss(features_df, training_end_date):
    df = features_df.copy()
    n = training_end_date
    train = df.loc[:n]
    test = df.loc[n:]
    X_train = train.drop(columns="Close").values
    y_train = train[["Close"]].values
    X_test = test.drop(columns="Close").values
    y_test = test[["Close"]].values
    test_date_index = test.index
    
    return X_train, y_train, X_test, y_test, test_date_index
#==================== END ==================================
