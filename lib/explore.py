from pandas import DataFrame


def explore(data: DataFrame):
    # get first few rows
    print(data.head())
    # get summary
    print(data.info())
    # get descriptive statistics
    print(data.describe())
