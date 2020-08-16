from tensorflow.random import set_seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop

# importing utility
from featureEngineering import etl, weather_features, commodity_features
from modelDesignUtility import RNN_input, RNN_preprocessing, multi_step_plot, plot_train_history

# RNN Constants
set_seed(88)
EVALUATION_INTERVAL = 200
EPOCHS = 5


'''
Commodity Price Predictions:

Commodities are typically driven by supply (volume) and demand (sentiment)
'''

def predict_temperature_forward(dataset):
    '''
    Multi step prediction model for average temperature using
    - Rainfall (mm)
    - Maximum wind gust (as Wx, Wy vectors)
    - 9am relative humidity ()
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
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



def predict_corn_forward(dataset):
    '''
    Multi step prediction model for CORN
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    '''
    TRAIN_SPLIT = 300
    PAST_HISTORY = 50
    FUTURE_TARGET = 10
    
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
    '''

    TRAIN_SPLIT = 300
    PAST_HISTORY = 50
    FUTURE_TARGET = 10
    
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


def sentiment_analysis(d: dict):
    '''
    Sentiment analysis for Commodity news
    '''

    pass



def main():

    # Weather
    #  avg_temp = weather_features("Weather.csv", ['AvgTemp', 'Rainfall (mm)', '9am relative humidity (%)', 'max Wx', 'max Wy'])
    #  predict_temperature_forward(avg_temp)

    # Corn
    #  corn_price = commodity_features("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    #  predict_corn_forward(corn_price)

    # Wheat
    #  wheat_price = commodity_features("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    #  predict_wheat_forward(normalise_train_data(wheat_price))




if __name__ == '__main__':
    main()
