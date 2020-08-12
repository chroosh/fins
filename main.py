import pandas as pd
import matplotlib.pyplot as plt

'''
Feature Engineering

more etl
- xlsx conversion to csv, removing graphs
- convert open interest in commodities.xlsx to remove comma separator
- convert
- dropna used to remove missing values

todo
- station 3: model design
- station 4: implementation
- station 5: the endgame xD
'''

files = "files/"



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



def plotNormalised(features):
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


'''
for commodities, we look at continuous price "last" and the volume "open interest" and plot them separately for feature engineering purposes

however, beacuse of the differing order of magnitudes bewteen the features, they can't be plotted on the same graph for comparison. the data needs to be normalised

this is essential for the extraction of seasonlity and "unwanted" deviations for RNN modelling
'''
def commodityFeatures(commodity: str):
    '''
    Feature engineering (continuous price and volume) for given corn and wheat commodities
    '''

    df = etl(commodity)

    features_considered = ['Last', 'Open Interest']
    features = df[features_considered]

    print("Plotting commodities")
    plotFeatures(features, df['Date'])
    plotNormalised(features)


'''
for weather, we look at the average temperature and the rainfall

again we need to normalise to visualise both features on the same graph
'''
def weatherFeatures():
    '''
    Feature engineering (average temperature and rainfall) for given weather data
    '''

    weatherDF = etl("Weather.csv")

    weatherDF['AvgTemp'] = (weatherDF['Minimum temperature (C)'] + weatherDF['Maximum temperature (C)']) / 2
    features_considered = ['AvgTemp', 'Rainfall (mm)']
    weatherFeatures = weatherDF[features_considered]

    print("Plotting weather")
    plotFeatures(weatherFeatures, weatherDF['Date'])
    plotNormalised(weatherFeatures)


def main():

    # Station 2: Feature Engineering
    commodityFeatures("WHEAT_pricehistory.csv")
    commodityFeatures("CORN_pricehistory.csv")
    weatherFeatures()

    # Station 3: Model Design

if __name__ == "__main__":
    main()
