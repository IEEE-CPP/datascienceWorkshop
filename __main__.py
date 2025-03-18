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
from scipy.stats import pearsonr
from ydata_profiling import ProfileReport

from lib.chartSpecificData import survivalFrame
from lib.clean import cleanAge, cleanEmbarked, cleanFare, dropIrrelevant
from lib.featureEngineering import addFamilyCountData
from lib.numericConversion import (SexConversion, embarkedConverter,
                                   sexConverter)

# %% [markdown]
"""
### matplotlib options

This is just because my local setup is weird
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

processedData = compose(cleanData, convertDataToNumeric)(data)
processedDataRows = len(processedData)

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

# %% [markdown]
"""
### Survival by Sex
"""


# %%

survivalFrame("Sex", processedData).rename(columns={1: "Female", 0: "Male"}).plot.bar()
plt.show()

# %% [markdown]
"""
Survival by class
"""

# %%
pclass = survivalFrame("Pclass", processedData).sort_index(axis=1)
pclass.plot.bar()
plt.show()

# %% [markdown]
"""
Survival by age
"""
# %%
survivalFrame("Age", processedData).sort_index(axis=1).transpose().plot.line()
plt.show()

# %% [markdown]
"""
Survival of women by age
"""

# %%
survivalFrame("Age", processedData[processedData["Sex"] == 1]).sort_index(
    axis=1
).transpose().plot.line()
plt.show()

# %% [markdown]
"""
Survival of men by age
"""

# %%
survivalFrame("Age", processedData[processedData["Sex"] == 0]).sort_index(
    axis=1
).transpose().plot.line()
plt.show()

# %% [markdown]
"""
Survival of children by age
"""

# %%
survivalFrame("Age", processedData[processedData["Age"] < 18]).sort_index(
    axis=1
).transpose().plot.line()
plt.show()

# %%
survivalFrame("Age", processedData[processedData["Age"] < 18]).sort_index(
    axis=1
).transpose().plot.bar()
plt.show()

# %% [markdown]
"""
Total survival of all children
"""

# %%
survivalFrame("Age", processedData[processedData["Age"] < 18]).sort_index(
    axis=1
).transpose().sum().plot.bar()
plt.show()

# %%
survivalFrame("Age", processedData[processedData["Age"] < 18]).sort_index(
    axis=1
).transpose().sum()

# %% [markdown]
"""
The obvious conclusion is that being a child doesn't help your chancesa all that much
"""

# %%
data

# %%
processedData

# %%
