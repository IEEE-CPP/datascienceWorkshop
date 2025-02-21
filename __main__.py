# %% Imports and libraries [markdown]
"""
# Imports and library functions
"""
# %%
from dataclasses import dataclass
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from ydata_profiling import ProfileReport

from lib.clean import cleanAge, cleanEmbarked, cleanFare, dropIrrelevant
from lib.featureEngineering import addFamilyCountData
from lib.numericConversion import embarkedConverter, sexConverter

# %% [markdown]
"""
### matplotlib options
"""

# %%
matplotlib.use("ipympl")
plt.ioff()

# %% [markdown]
"""
## Compose
a function that enables functional composition
compose :: function, function, ... -> function

compose(f, g, h, i)(x) is equivalent to i(h(g(f(x))))
"""


# %%
def compose(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs, lambda x: x)


# %% [markdown]
# ## Import The Data

# %%

data: DataFrame = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
)

# %% Data Exploration [markdown]
"""
## Data Exploration part 1
"""
# %% [markdown]
"""
### Get the first few rows
"""
# %%
data.head()
# %% [markdown]
"""
### Get the last few rows
"""
# %%
data.tail()

# %% [markdown]
"""
### Get summary, data types, that sort of thing
"""

# %%
data.info()

# %% [markdown]
"""
### Get descriptive statistics
"""

# %%
data.describe()

# %% [markdown]
"""
### Get number of nulls
"""

# %%
data.isnull().sum()


# %% [markdown]
"""
### ydataprofiler
"""
# %%
# ProfileReport(data, title="Titanic Profiling Report")

# %% data Preprocessing [markdown]
"""
# Data Preprocessing

Here we are just getting rid of null values and dropping irrelevant data that we don't need
"""

# %%
cleanData = compose(cleanAge, cleanFare, cleanEmbarked, dropIrrelevant)
convertDataToNumeric = compose(sexConverter, embarkedConverter)

processedData = compose(cleanData, convertDataToNumeric, addFamilyCountData)(data)

# %% [markdown]
"""
# Data Exploration Part II and Data Visualization
"""
# %% [markdown]
"""
### ydata profiler again because catagorical data sucks
"""
# %%
# ProfileReport(processedData, title="Titanic Profiling Report")

# %% [markdown]
"""
### Correlation Graph
"""

# %%
processedDataCorrMatrix = processedData.corr().round(3)
sns.heatmap(processedDataCorrMatrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
