# %% Imports and libraries [markdown]
"""
## Imports and library functions
"""
# %%
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

# %% [markdown]
"""
### Compose
a function that enables functional composition
compose :: function, function, ... -> function

compose(f, g, h, i)(x) is equivalent to i(h(g(f(x))))
"""


# %%
def compose(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs, lambda x: x)


# %% data preprocessing [markdown]
"""
# Data Preprocessing
"""

# %%
data: DataFrame = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
)

cleanData = compose(cleanAge, cleanFare, cleanEmbarked, dropIrrelevant)
convertDataToNumeric = compose(sexConverter, embarkedConverter)

processedData = compose(cleanData, convertDataToNumeric, addFamilyCountData)(data)

print(processedData)
print(data)
explore(data)
