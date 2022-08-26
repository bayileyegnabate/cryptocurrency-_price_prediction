# Cryptocurrency Price Prediction

## Objective:  
- Having the ability put in any crypto currency and predict next day price 
- Assist traders in making an informed decisions about crytoprices in the short-term


## Resources
**Data Sources:**
- Kaggle:cryptocurrency_price_prediction
- Yahoo Finance -(historical data)

**Software:**
- Python
- SQLite
- Amazon Web Services: Database

## Overview:
As a team of four, we decided to choose 11 coins to predict next day price. The data from Yahoo Finance was parsed in jupyter notebook to show the date, closing price, volume, and ticker. See image below:

<img width="335" alt="data_gathering" src="https://user-images.githubusercontent.com/100165760/186288464-55ff5d6c-8c7c-45b8-b225-b9c6a90960dd.png">


We worked with two different machine learning models (lstm and xgboost) to decifer which model would best demostrate the next day coin price accurately.

## Results:

### LSTM
The model was trained to use the previous 5 days of data to predict what the coin price would be for the next day. 
Each coin dataframe was plotted on a graph: 

<img width="402" alt="plot_graph_avax" src="https://user-images.githubusercontent.com/100165760/186289665-5eaa3213-2222-4f07-8cae-36dcbfdda0f5.png">



After the coin was split, trained, tested, scaled, and reshaped, the model was built.

<img width="402" alt="model_prediction" src="https://user-images.githubusercontent.com/100165760/186290032-02fa774c-c850-46bc-8e03-ea49f73dbf3a.png">

### XGBOOST
The same cryptocurrency coins were utilized in the XGBoost model. Each coin was transferred to a dataframe and then trained and fitted. The image below shows the trained XGBoost visual of AVAX-Avalance-USD:

<img width="699" alt="avax_predicted_xgboost" src="https://user-images.githubusercontent.com/100165760/186677632-f13a3c6b-f3e7-494a-a5f2-af62e5eec088.png">

## Conclusion:

After comparing the data from the two machine learning models, we chose to use the XGBoost model. XGBoost was able to improve speed, and model performance. 

In order to build a more robust, accurate model, in the future we should consider additional factors like, volume, the coins high and low price, circulating supply, percent change over 7 days and market cap in order to parse out factors that may have a large influence on the predicting the price tomorrow.


Limiting Factors
 
We recognize that in order for models to most accurate on a time series, we must see some type of pattern in the data. Because of this, we anticipate this model to best predict cryptocurrency with many data points, preferably many years old. Since Bitcoin is one of the oldest in this space, we have sufficient data to estimate price prediction for bitcoin, however, coins that may be newer have not produced pattern enough to give a reliable prediction.


