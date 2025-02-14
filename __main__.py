from dataclasses import dataclass
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from lib.clean import cleanAge, cleanEmbarked, cleanFare, dropIrrelevant
from lib.explore import explore
from lib.featureEngineering import addFamilyCountData
from lib.numericConversion import embarkedConverter, sexConverter


def compose(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs, lambda x: x)


if __name__ == "__main__":
    data: DataFrame = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
    )

    cleanData = compose(cleanAge, cleanFare, cleanEmbarked, dropIrrelevant)
    convertDataToNumeric = compose(sexConverter, embarkedConverter)

    processedData = compose(cleanData, convertDataToNumeric, addFamilyCountData)(data)

    print(processedData)
    print(data)
