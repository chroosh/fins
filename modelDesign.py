from tensorflow.random import set_seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop

# importing utility
from featureEngineering import etl, weather_features, commodity_features
from modelDesignUtility import RNN_input, RNN_preprocessing, multi_step_plot, plot_train_history, normalise_train_data, FUTURE_TARGET

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
    Multi step prediction model for average temperature
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    '''

def predict_corn_forward(dataset):
    '''
    Multi step prediction model for CORN
    
    :param pandas.DataFrame dataset: Dataset to be used, target in first column
    '''
    
    RNN_inputs = RNN_preprocessing(dataset, True)

    model = Sequential()
    model.add(LSTM(16, input_shape=RNN_inputs.input_shape))
    model.add(Dropout(0.4))
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
    
    RNN_inputs = RNN_preprocessing(dataset, True)

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



def main():

    # Weather
    #  avg_temp = weather_eatures("Weather.csv", ['AvgTemp', 'Rainfall (mm)'])


    # Corn
    #  corn_price = commodity_features("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    #  predict_corn_forward(normalise_train_data(corn_price))

    # Wheat
    wheat_price = commodity_features("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    print(wheat_price)
    predict_wheat_forward(normalise_train_data(wheat_price))

    pass



if __name__ == '__main__':
    main()
