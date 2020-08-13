import pandas as pd
import matplotlib.pyplot as plt

'''
Feature Engineering


issues:
- duplicate featureEngineering.plotNormalised and modelDesign.normaliseData

todo:
- station 2: feature engineering:
    - more etl (file i/o and dropna)
    - plotting utility (standard and normalisedkj)
- station 3: model design
    - TODO model utility
    - TODO actual models for commodities, weather, cash flow
    - TODO refactor model utility after implementing models
- station 4: implementation
    - models currently only predicts the next 5 values out of a training subsample that you already know  
    - somehow we need to setup and train our model, and then save the "parameters" to run on the last 50 to predict the next 5
- station 5: the endgame xD
    - lol

- sentiment analysis
- report
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

def commodityFeatures(commodityFile: str, features_considered: []) -> pd.DataFrame:
    '''
    Feature engineering (continuous price and volume) for given corn and wheat commodities

    :param string commodityFile: Filename containing raw data
    :return pandas.DataFrame features
    '''

    df = etl(commodityFile)
    features = df[features_considered]

    print("Plotting commodities")
    plotFeatures(features, df['Date'])
    plotNormalised(features)

    return features


def weatherFeatures(weatherFile: str, features_considered: []) -> pd.DataFrame:
    '''
    Feature engineering (average temperature) for given weather data

    :param string weatherFile: Filename containing raw data
    :return pandas.DataFrame weatherFeatures: The pandas df containing selected weather features
    '''

    df = etl(weatherFile)

    # Given min/max temperature, take average
    df['AvgTemp'] = (df['Minimum temperature (C)'] + df['Maximum temperature (C)']) / 2
    features = df[features_considered]

    print("Plotting weather")
    plotFeatures(features, df['Date'])
    plotNormalised(features)

    return features


def main():

    # Station 2: Feature Engineering
    commodityFeatures("WHEAT_pricehistory.csv", ['Last', 'Open Interest'])
    commodityFeatures("CORN_pricehistory.csv", ['Last', 'Open Interest'])
    weatherFeatures("Weather.csv", ['AvgTemp'])

    # Station 3: Model Design

if __name__ == "__main__":
    main()
