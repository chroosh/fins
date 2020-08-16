# FINS Assignment
_I lowkey dont know what I'm doing_
_Finished the assignment, still don't know what I'm doing_

## Tasks
- Predict future commodity prices of wheat and corn ("WHEAT_PriceHistory.xlsx", "CORN_PriceHistory.xlsx")
- Predict the future average temperature given weather data ("Weather.csv")
- Predict future cash flows of a client given client cash account data ("Client_Cash_Accounts.xlsx")
- Perform sentiment analysis on "Commodity.json"


## Aim
The aim of this is to build a "financial product" for a neobank with agricultural clients. I guess the product could vary from simple predictive dashboards to more complex insurance/hedging products for long(er?) term predictions.

## Wiki link:
https://www.notion.so/RNN-LSTM-7fa6e0b154ed4f54b08c0e315f0af6a7

## Progress

Issues:
- duplicate featureEngineering.plotNormalised and modelDesign.normaliseData

Todo:
- station 2: feature engineering:
    - DONE more etl utility (file i/o and dropna)
    - DONE plotting utility (standard and normalised)

		- feature engineering
			- DONE commodities
			- DONE weather
			- DONE cash flow
			- DONE sentiment

- station 3: model design
    - DONE model utility (ported from JH code)
    - DONE basic LSTM RNN
    - TODO specific models for 
			- commodities
				- ISH feature selection, prereq: sentiment
				- ISH hyperparameter tuning
			- weather
				- DONE feature selection
				- ISH hyperparameter tuning
			- sentiment
				- DONE analysis
			- cash flow
				- feature selection, prereq: commodity, weather, sentiment
					-  how do you use other models to do this?
				- hyperparameter tuning
    - DONE refactor model utility after implementing models

- station 4: implementation
	- model needs to predict next 10 values after last date in dataset
		- commodities
		- weather
		- cash flow

- station 5: the endgame xD
  - building the actual product

- DONE report
