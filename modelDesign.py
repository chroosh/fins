import numpy as np
from tensorflow.random import set_seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop

# importing utility
from featureEngineering import etl, weather_features, commodity_features, cash_features, sentiment_features
from modelDesignUtility import RNN_input, RNN_preprocessing, multi_step_plot, plot_train_history, show_plot


# RNN Constants
set_seed(88)
EVALUATION_INTERVAL = 200
EPOCHS = 5

'''
Weather Average Temperature Predictions:

Using the following factors:
- Rainfall (mm)
- Maximum wind gust (as Wx, Wy vectors)
- 9am relative humidity (%)
'''
def predict_temperature_forward(dataset):
    '''
    Multi step prediction model for average temperature using
   
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    :return tensorflow.keras.Model model: Trained temperature prediction model
    '''

    TRAIN_SPLIT = 200
    PAST_HISTORY = 30
    FUTURE_TARGET = 10

    RNN_inputs = RNN_preprocessing(dataset, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, True)

    model = Sequential()
    model.add(LSTM(8, input_shape=RNN_inputs.input_shape))
    # maybe add another layer?
    model.add(Dropout(0.7))
    model.add(Dense(FUTURE_TARGET))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae', metrics=['acc'])

    history = model.fit(RNN_inputs.train_data, 
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=RNN_inputs.val_data,
                        validation_steps=50)

    for x, y in RNN_inputs.val_data.take(3):
        # probably holds the key to make it predict
        multi_step_plot(x[0], y[0], model.predict(x)[0])
        
    plot_train_history(history, 'Multi-Step Training and validation loss')


'''
Commodity Price Predictions:

Commodities are typically driven by supply (volume) and demand (sentiment)
'''
def predict_corn_forward(dataset):
    '''
    Multi step prediction model for CORN
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    :return tensorflow.keras.Model model: Trained corn price prediction model
    '''
    TRAIN_SPLIT = 300
    PAST_HISTORY = 60
    FUTURE_TARGET = 30
    
    RNN_inputs = RNN_preprocessing(dataset, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, True)

    model = Sequential()
    model.add(LSTM(16, input_shape=RNN_inputs.input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(FUTURE_TARGET))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae', metrics=['acc'])

    history = model.fit(RNN_inputs.train_data, 
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=RNN_inputs.val_data,
                        validation_steps=50)

    for x, y in RNN_inputs.val_data.take(3):
        # probably holds the key to make it predict
        multi_step_plot(x[0], y[0], model.predict(x)[0])
        
    plot_train_history(history, 'Multi-Step Training and validation loss')


def predict_wheat_forward(dataset):
    '''
    Multi step prediction model for WHEAT
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    :return tensorflow.keras.Model model: Trained wheat price prediction model

    '''

    TRAIN_SPLIT = 300
    PAST_HISTORY = 60
    FUTURE_TARGET = 30
    
    RNN_inputs = RNN_preprocessing(dataset, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, True)

    model = Sequential()
    model.add(LSTM(12, input_shape=RNN_inputs.input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(FUTURE_TARGET))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae', metrics=['acc'])

    history = model.fit(RNN_inputs.train_data, 
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=RNN_inputs.val_data,
                        validation_steps=50)

    for x, y in RNN_inputs.val_data.take(3):
        multi_step_plot(x[0], y[0], model.predict(x)[0])
    plot_train_history(history, 'Multi-Step Training and validation loss')

    return model


'''
Sentiment Analysis:
Using NLTK Vader
'''
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def sentiment_analysis(dataset):
    '''
    Sentiment analysis for Commodity news

    :param pandas.DataFrame dataset: Dataset to be used
    :return pandas.DataFrame dataset: Dataset with sentiment scores
    '''
    dataset['Scores'] = dataset['Headline'].apply(lambda headline: sid.polarity_scores(headline))

    return dataset


'''
Cash flow prediction
'''
def predict_cash_forward(dataset):
    '''
    Multi step prediction model for Cash flow
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    :return tensorflow.keras.Model model: Trained cash flow prediction model
    '''

    TRAIN_SPLIT = 60
    PAST_HISTORY = 20
    FUTURE_TARGET = 10
    
    RNN_inputs = RNN_preprocessing(dataset, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET)

    model = Sequential()
    model.add(LSTM(3, input_shape=RNN_inputs.input_shape))
    model.add(Dropout(0.8))
    model.add(Dense(FUTURE_TARGET))

    model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae', metrics=['acc'])

    history = model.fit(RNN_inputs.train_data, 
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=RNN_inputs.val_data,
                        validation_steps=50)

    #  for x, y in RNN_inputs.val_data.take(3):
    #      plot = show_plot([x[0].numpy(), y[0].numpy(),
    #                       model.predict(x)[0]], 0, 'Simple LSTM model')
        
    plot_train_history(history, 'Multi-Step Training and validation loss')

    return model


def main():

    # Weather
    #  avg_temp = weather_features("Weather.csv", ['AvgTemp', 'Rainfall (mm)', '9am relative humidity (%)', 'max Wx', 'max Wy'])
    #  predict_temperature_forward(avg_temp)

    # Corn
    corn_price = commodity_features("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    predict_corn_forward(corn_price)

    # Wheat
    wheat_price = commodity_features("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    predict_wheat_forward(normalise_train_data(wheat_price))


    # Sentiment
    commodity_news = sentiment_features("Commodity_News.json", ['Headline'])
    sentiment_analysis(commodity_news)

    # Cash flows
    #  cash_flows = cash_features("Client_Cash_Accounts.xlsx", ['1x', '2x', '3x', '4x', '5x'], ['Cash Balance'])
    #  for client in cash_flows:
    #      predict_cash_forward(cash_flows[client])




if __name__ == '__main__':
    main()
