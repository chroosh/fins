import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# importing utility functions from main
from featureEngineering import etl, weatherFeatures, commodityFeatures

# Constants (temporary, these might need to change based on RNN model so they should be specified by the model, and not outside the scope like here)
TRAIN_SPLIT = 300
tf.random.set_seed(88)
EVALUATION_INTERVAL = 200
EPOCHS = 5
BATCH_SIZE = 50
BUFFER_SIZE = 1000

past_history = 50
future_target = 10
STEP = 1

# Utility functions
#  x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 #  TRAIN_SPLIT, past_history,
                                                 #  future_target, STEP)
def multivariate_data(dataset, target, start, end, history_size, target_size, step, single_step=False):
    '''
    Multivariate_data
    TODO - Purpose unknown

    :param pandas.DataFrame.values dataset: Normalised values of dataset
    :param pandas.DataFrame.values[:, x] target: Target column of dataset to predict
    :param int start: Start value
    :param int end: End value
    :param int history_size: Size of past history
    :param int target_size: Size of future target to predict
    :param int step: Step ?
    :param bool single_step: ?

    '''

    data = []
    labels = []
    start += history_size
    if end is None:
        end = len(dataset) - target_size

    for i in range(start, end):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def baseline(history):
    return np.mean(history)

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt.show()

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def normaliseData(features) -> pd.DataFrame:
    '''
    This is duplicated from featureEngineering.py
    '''
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    return dataset


class RNNinput:
    '''
    RNNinput Class
    Used to store input parameters for LSTM RNN

    :param tensorflow.data.Dataset train_data: Training Data
    :param tensorflow.data.Dataset val_data: Validation Data
    :param ? input_shape: Input shape of the LSTM RNN
    '''
    def __init__(self, train_data, val_data, input_shape):
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape


def RNNpreprocessing(dataset, plot=False):
    '''
    Preprocessing for RNN using multivariate data
    Target i
    TODO - abbreviate function calls

    :param pandas.DataFrame.values dataset: Normalised values of dataset
    :param bool plot: Set to true to plot dataset, false otherwise
    '''

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)


    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    if plot:
        for x, y in train_data_multi.take(1):
            multi_step_plot(x[0], y[0], np.array([0]))

    return RNNinput(train_data_multi, val_data_multi, x_train_multi.shape[-2:])



'''
Commodity Price Predictions:

Commodities are typically driven by factors such as:
- Supply and demand
- Value of the US dollar (as most commodities are priced in US dollars)

TODO
- setting up multivariate data code is the same for both Corn and Wheat
- abbreviate function calls


'''
def predictCornForward(dataset):
    '''
    Multi step prediction model for CORN
    
    :param pandas.DataFrame dataset: Dataset to be used to predict forward
    '''
    
    RNNinputs = RNNpreprocessing(dataset, True)

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(10, input_shape=RNNinputs.input_shape))
    multi_step_model.add(tf.keras.layers.Dropout(0.2))
    multi_step_model.add(tf.keras.layers.Dense(future_target))
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), 
                             loss='mae', metrics=['acc'])

    multi_step_history = multi_step_model.fit(RNNinputs.train_data, 
                                              epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=RNNinputs.val_data,
                                              validation_steps=50)

    for x, y in RNNinputs.val_data.take(3):
        # probably holds the key to make it predict
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
        
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


def predictWheatForward(dataset):
    '''
    Multi step prediction model for WHEAT
    
    :param pandas.DataFrame dataset: Dataset to be used to predict forward with target attribute in first column
    '''
    
    RNNinputs = RNNpreprocessing(dataset, True)

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(16, input_shape=RNNinputs.input_shape))
    multi_step_model.add(tf.keras.layers.Dropout(0.9))
    multi_step_model.add(tf.keras.layers.Dense(future_target))


    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), 
                             loss='mae', metrics=['acc'])

    multi_step_history = multi_step_model.fit(RNNinputs.train_data, 
                                              epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=RNNinputs.val_data,
                                              validation_steps=50)

    for x, y in RNNinputs.val_data.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')



def main():
    print("Train split: " + str(TRAIN_SPLIT))
    print("Past history used: " + str(past_history))

    # Weather
    #  avgTemp = weatherFeatures("Weather.csv", ['AvgTemp', 'Rainfall (mm)'])
    #  predictFivePtForward(normaliseData(avgTemp))


    # Corn
    #  cornPrice = commodityFeatures("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    #  predictCornForward(normaliseData(cornPrice))

    # Wheat
    wheatPrice = commodityFeatures("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    predictCornForward(normaliseData(wheatPrice))

    pass



if __name__ == '__main__':
    main()
