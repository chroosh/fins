import pandas as pd
import matplotlib.pyplot as plt

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
    
    :param string filename: The name of the data file to be opened
    :return pandas.DataFrame df: The dataframe that the data file has been loaded into
    '''
    df = pd.read_csv(files + filename)
    df.dropna(
        axis = 0,
        how = "any",
        thresh = None,
        subset = None,
        inplace = True
    )
    return df


def plotFeatures(features, index):
    '''
    Plot the values of a features

    :param pandas.DataFrame features: The features to be plotted
    :param pandas.DataFrame index: The date over which features are plotted
    '''
    features.index = index
    print(features.head())
    features.plot(subplots=True)
    plt.show()


def plotNormalised(features)-> pd.DataFrame.values:
    '''
    Plots the normalised values of features

    :param pandas.DataFrame features: The features to be normalised
    :return pandas.DataFrame.values dataset: Normalised value of dataset
    '''
    dataset = features.values
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    print(data_mean)
    print(data_std)
    dataset = (dataset-data_mean)/data_std
    print(dataset[:5])
    plt.plot(dataset)
    plt.show()

    return dataset

def commodityFeatures(commodityFile: str, featuresConsidered: [], plot=False) -> pd.DataFrame:
    '''
    Feature engineering (continuous price and volume) for given corn and wheat commodities

    :param string commodityFile: Filename containing raw data
    :param array featuresConsidered: Array containing features to be considered
    :param bool plot: Set to true to plot features, false otherwise

    :return pandas.DataFrame features
    '''

    df = etl(commodityFile)
    features = df[featuresConsidered]

    if plot:
        print("Plotting commodities")
        plotFeatures(features, df['Date'])
        plotNormalised(features)

    return features


def weatherFeatures(weatherFile: str, featuresConsidered: [], plot=False) -> pd.DataFrame:
    '''
    Feature engineering (average temperature) for given weather data

    :param string weatherFile: Filename containing raw data
    :param array featuresConsidered: Array containing features to be considered
    :param bool plot: Set to true to plot features, false otherwise

    :return pandas.DataFrame weatherFeatures: The pandas df containing selected weather features
    '''

    df = etl(weatherFile)

    # Given min/max temperature, take average
    df['AvgTemp'] = (df['Minimum temperature (C)'] + df['Maximum temperature (C)']) / 2
    features = df[featuresConsidered]

    if plot:
        print("Plotting weather")
        plotFeatures(features, df['Date'])
        plotNormalised(features)

    return features


# TODO
def cashFeatures():
    pass


# TODO should not be run as a standalone
def main():

    # Station 2: Feature Engineering
    commodityFeatures("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    commodityFeatures("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    weatherFeatures("Weather.csv", ['AvgTemp'])

if __name__ == "__main__":
    main()
