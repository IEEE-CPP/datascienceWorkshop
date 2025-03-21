# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% Imports and libraries [markdown]
"""
# Imports and library functions
"""
# %%
# %load_ext autoreload
# %autoreload 2

# %%
from dataclasses import dataclass
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import plot_tree
from ydata_profiling import ProfileReport

from lib.chartSpecificData import survivalFrame
from lib.clean import cleanAge, cleanEmbarked, cleanFare, dropIrrelevant
from lib.featureEngineering import addFamilyCountData, addIsAdult
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
# Data Exploration part 1
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

# CCXZX%% data Preprocessing [markdown]
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
