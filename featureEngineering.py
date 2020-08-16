import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

'''
Feature Engineering

Provides utility for feature engineering:
- File I/O
- Feature plotting and normalisation
'''

files = "files/"


'''
ETL precondition
- xlsx converted to csv, remove graphs and move data to top left cell
- modified 'Open Interest' column in 'WHEAT_pricehistory.csv' and 'CORN_pricehistory.csv' files to remove comma separator
- modified 'Minimum temperature (C)' and 'Maximum temperature (C)' in 'Weather.csv' file to remove degree character
'''
def etl(filename: str) -> pd.DataFrame:
    '''
    Opens filename and drops invalid values
    Compatible with csv, (multi-page) xlsx, and json
    Note: untested for single-page xlsx
    
    :param string filename: The name of the data file to be opened
    :return pandas.DataFrame df: Dataframe containing loaded and cleaned data (csv or json)
    :return dict d: Dictionary containing loaded and cleaned dataframes for each page (xlsx)
    '''

    def clean(df):
        df.dropna(
            axis = 0,
            how = "any",
            thresh = None,
            subset = None,
            inplace = True
        )

    filetype = filename.split(".")[-1]

    if filetype == "csv":
        df = pd.read_csv(files + filename)
        clean(df)
        return df
    elif filetype == "xlsx":
        d = pd.read_excel(files + filename, sheet_name=None)
        for each in d:
            clean(d[each])
        return d
    elif filetype == 'json':
        df = pd.read_json(files + filename)
        return df
        

def plot_features(features):
    '''
    Plot the values of a features

    :param pandas.DataFrame features: The features to be plotted
    '''
    # print(features.head())
    features.plot(subplots=True)
    plt.show()


def plot_normalised(features)-> pd.DataFrame.values:
    '''
    Plots the normalised values of features

    :param pandas.DataFrame features: The features to be normalised
    :return pandas.DataFrame.values dataset: Normalised value of dataset
    '''
    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    # print(data_mean)
    # print(data_std)
    dataset = (dataset-data_mean)/data_std
    # print(dataset[:5])
    plt.plot(dataset)
    plt.show()

    return dataset

def commodity_features(commodity_file: str, features_considered: [], plot=False) -> pd.DataFrame:
    '''
    Feature engineering (continuous price and volume) for given corn and wheat commodities

    :param string commodity_file: Filename of file containing raw data
    :param array features_considered: Array containing features to be considered
    :param bool plot: Set to true to plot features, false otherwise

    :return pandas.DataFrame features
    '''

    df = etl(commodity_file)
    features = df[features_considered]
    features.index = df['Date']

    if plot:
        print("Plotting commodities")
        plot_features(features)
        plot_normalised(features)

    return features


to_degrees = {
    'N': 0,
    'NNE': 22.5,
    'NE': 45,
    'ENE': 67.5,
    'E': 90,
    'ESE': 112.5,
    'SE': 135,
    'SSE': 157.5,
    'S': 180,
    'SSW': 202.5,
    'SW': 225,
    'WSW': 247.5,
    'W': 270,
    'WNW': 292.5,
    'NW': 315,
    'NNW': 337.5
}

def weather_features(weather_file: str, features_considered: [], plot=False) -> pd.DataFrame:
    '''
    Feature engineering (average temperature) for given weather data

    :param string weather_file: Filename of file containing raw data
    :param array features_considered: Array containing features to be considered
    :param bool plot: Set to true to plot features, false otherwise

    :return pandas.DataFrame weatherFeatures: The pandas df containing selected weather features
    '''

    df = etl(weather_file)

    # Given min/max temperature, take average
    df['AvgTemp'] = (df['Minimum temperature (C)'] + df['Maximum temperature (C)']) / 2
    
    # Calculate wind Wx, Wy
    wd_deg = np.array([to_degrees[direction] for direction in df['Direction of maximum wind gust']])
    wd_rad = wd_deg * np.pi / 180

    df['max Wx'] = df['Speed of maximum wind gust (km/h)']* np.cos(wd_rad)
    df['max Wy'] = df['Speed of maximum wind gust (km/h)']* np.sin(wd_rad)

    features = df[features_considered]
    features.index = df['Date']

    if plot:
        print("Plotting weather")
        plot_features(features)
        plot_normalised(features)

    return features


def cash_features(cash_file: str, pages_considered: [], features_considered: [], plot=False) -> dict:
    '''
    Feature engineering for given cash flow

    :param string cash_file: Filename of file containing raw data
    :param features_considered: Array containing features to be considered
    :param bool plot: Set to true to plot features, false otherwise

    :return dict features: Dictionary containing features of each client
    '''
    d = etl(cash_file)

    # select only features considered from pages considered
    features = {}
    for each in pages_considered:
        #  print(d[each][features_considered])
        features[each] = d[each][features_considered]
        features[each].index = d[each]['Date']
        if plot:
            print("Plotting cash")
            plot_features(features[each])
            plot_normalised(features[each])

    return features


def sentiment_features(sentiment_file: str, features_considered: []):
    '''
    Feature engineering for given commodity news

    :param string sentiment_file: Filename of file containing raw data
    :return dict d: json-ified sentiment file
    '''
    df = etl(sentiment_file)
    
    features = df[features_considered]
    features.index = df['Date']
    
    return features


# Testing only
def main():

    # Station 2: Feature Engineering
    #  commodity_features("WHEAT_pricehistory.csv", ['Last', 'Open Interest'], True)
    #  commodity_features("CORN_pricehistory.csv", ['Last', 'Open Interest'], True)
    #  weather_features("Weather.csv", ['AvgTemp', 'Rainfall (mm)', 'max Wx', 'max Wy', '9am relative humidity (%)'], True)
    #  cash_features("Client_Cash_Accounts.xlsx", ['1x', '2x', '3x', '4x', '5x'], ['Cash Balance'], True)
    #  sentiment_features("Commodity_News.json", ['Headline'])

    pass

if __name__ == "__main__":
    main()
