# FINS Assignment
_I lowkey dont know what I'm doing_

## Tasks
- Predict future commodity prices of wheat and corn ("WHEAT_PriceHistory.xlsx", "CORN_PriceHistory.xlsx")
- Predict the future average temperature given weather data ("Weather.csv")
- Predict future cash flows of a client given client cash account data ("Client_Cash_Accounts.xlsx")
- Perform sentiment analysis on "Commodity.json"

The aim of this is to build a "financial product" for a neobank with agricultural clients. I guess the product could vary from simple predictive dashboards to more complex insurance/hedging products for long(er?) term predictions.


## Notes
LSTM RNN's can be used to predict future commodity prices, average temperature and cash flow data. Something something about it being good with time series data.

Wiki link:
https://www.notion.so/RNN-LSTM-7fa6e0b154ed4f54b08c0e315f0af6a7

## Progress

Issues:
- duplicate featureEngineering.plotNormalised and modelDesign.normaliseData
- modularise code for stations

Todo:
- station 2: feature engineering:
    - DONE more etl (file i/o and dropna)
    - DONE plotting utility (standard and normalised)
- station 3: model design
    - DONE model utility
    - DONE basic universal model
    - TODO specific models for commodities, weather, cash flow
			- tuning features considered for model
      - hyperparameter tuning
    - TODO refactor model utility after implementing models
- station 4: implementation
    - models currently only predicts the next 5 values out of a training subsample that you already know  
    - somehow we need to setup and train our model, and then save the "parameters" to run on the last 50 to predict the next 5
- station 5: the endgame xD
    - building the actual product

- sentiment analysis
- report
