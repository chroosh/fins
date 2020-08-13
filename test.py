import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# importing utility functions from main
from main import etl, weatherFeatures, commodityFeatures

# Constants
TRAIN_SPLIT = 300
tf.random.set_seed(88)
EVALUATION_INTERVAL = 200
EPOCHS = 5
BATCH_SIZE = 50
BUFFER_SIZE = 1000

past_history = 50
STEP = 1

# Utility functions
def multivariate_data(dataset, target, start, end, history_size, target_size, step, single_step=False):
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
    This is duplicated
    '''
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    return dataset


def predictSinglePtForward(dataset):
    '''
    Single step prediction model - 1 step forward
    '''
    future_target = 5
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

    print ('Single window of past history : {}'.format(x_train_single[0].shape))

    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    # BUILD SINGLE POINT FORWARD LSTM RNN MODEL
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)

    plot_train_history(single_step_history,'Single Step Training and validation loss')

    for x, y in val_data_single.take(3):
        print(val_data_single)
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          single_step_model.predict(x)[0]], 5,
                         'Single Step Prediction')

def predictForward(dataset, future_target):
    '''
    Multi step prediction model

    :param pandas.DataFrame dataset: Dataset to be used to predict forward
    :param int future_target: How many steps to predict into the future
    '''
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))

    multi_step_model = tf.keras.models.Sequential()
    '''
    # how does the number of starting units affect the RNN prediction
    # how does the number of layers affect the RNN prediction
    # how does dropout affect RNN prediction
    '''
    multi_step_model.add(tf.keras.layers.LSTM(16, input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.Dropout(0.5))
    #  multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    #  multi_step_model.add(tf.keras.layers.Dropout(0.5))
    multi_step_model.add(tf.keras.layers.Dense(future_target))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50)

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


def main():
    print(TRAIN_SPLIT)
    print(past_history)
    #  avgTemp = weatherFeatures("Weather.csv", ['AvgTemp', 'Rainfall (mm)'])
    #  predictFivePtForward(normaliseData(avgTemp))


    cornPrice = commodityFeatures("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    predictForward(normaliseData(cornPrice), 5)
    #  wheatPrice = commodityFeatures("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])


    pass



if __name__ == '__main__':
    main()
