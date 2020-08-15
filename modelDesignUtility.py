from tensorflow.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data preprocessing constants
BATCH_SIZE = 50
BUFFER_SIZE = 1000
STEP = 1

TRAIN_SPLIT = 300
PAST_HISTORY = 50
FUTURE_TARGET = 10


# Utility functions
def multivariate_data(dataset, target, start, end, history_size, target_size, step, single_step=False):

    '''
    Multivariate_data

    :param pandas.DataFrame.values dataset: Normalised values of dataset
    :param pandas.DataFrame.values[:, x] target: Target column of to predict
    :param int start: Start value
    :param int end: End value
    :param int history_size: Size of past history
    :param int target_size: Size of future target to predict
    :param int step:
    :param bool single_step:


    :return np.array data: Return data
    :return np.array labels: Return labels
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


def normalise_train_data(features) -> pd.DataFrame:
    '''
    This is a soft duplicate of plot_normalised from featureEngineering.py
    '''
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    return dataset


class RNN_input:
    '''
    RNN_input Class
    Used to store input parameters for LSTM RNN

    :param tensorflow.data.Dataset train_data: Training Data
    :param tensorflow.data.Dataset val_data: Validation Data
    :param ? input_shape: Input shape of the LSTM RNN
    '''
    def __init__(self, train_data, val_data, input_shape):
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape


def RNN_preprocessing(dataset, plot=False):
    '''
    Preprocessing for RNN using multivariate data
    TODO - abbreviate function calls

    :param pandas.DataFrame.values dataset: Normalised values of dataset
    :param bool plot: Set to true to plot dataset, false otherwise
    '''

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                     TRAIN_SPLIT, PAST_HISTORY,
                                                     FUTURE_TARGET, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                 TRAIN_SPLIT, None, PAST_HISTORY,
                                                 FUTURE_TARGET, STEP)


    train_data_multi = Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    if plot:
        for x, y in train_data_multi.take(1):
            multi_step_plot(x[0], y[0], np.array([0]))

    return RNN_input(train_data_multi, val_data_multi, x_train_multi.shape[-2:])


