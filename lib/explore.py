from pandas import DataFrame


def explore(data: DataFrame):
    # get first few rows
    print(data.head())
    # get last few rows
    print(data.tail())
    # get summary, data types, that sort of thing
    print(data.info())
    # get descriptive statistics
    print(data.describe())
    # get number of nuls
    print(data.isnull().sum())
