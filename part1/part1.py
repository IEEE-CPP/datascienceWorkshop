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

# %% [markdown]
"""
# Data Science Part I
"""

# %% [markdown]
# """
# Imports and libs
# """
# from functools import partial, reduce
# from typing import List

# %%
import pandas as pd
import ydata_profiling
from pandas import DataFrame

# %%

data = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
)

# %%
data.info()

# %%
data.describe()

# %%
data.isnull().sum()
# %%
# ydata_profiling.ProfileReport(data, title="tisernta")


# %%
def compose(*funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs, lambda x: x)


# %%
def fillColumNaWithMedian(data: DataFrame, columnName: str) -> DataFrame:
    columnMedian = data[columnName].median()
    return data.fillna(value={columnName: columnMedian})


def fillColumNaWithMode(data: DataFrame, columnName: str) -> DataFrame:
    columnMedian = data[columnName].mode()
    return data.fillna(value={columnName: columnMedian})


def dataDropper(data: DataFrame, columns):
    return data.drop(columns, axis=1)


from functools import partial, reduce
from typing import List

cleanAge = partial(fillColumNaWithMedian, columnName="Age")
cleanFare = partial(fillColumNaWithMedian, columnName="Fare")
cleanEmbarked = partial(fillColumNaWithMode, columnName="Embarked")
dropIrrelent = partial(dataDropper, columns=["Name", "Ticket", "Cabin"])

# %%
from dataclasses import dataclass


@dataclass
class NumericConversionData:
    columnName: str
    conversionMap: dict[str, int]


SexConversion = NumericConversionData("Sex", {"male": 0, "female": 1})
EmbarkedConversion = NumericConversionData("Embarked", {"C": 0, "Q": 1, "S": 2})


def convertColToNumeric(
    data: DataFrame, columnData: NumericConversionData
) -> DataFrame:
    convertedData = data.copy()
    convertedData[columnData.columnName] = data[columnData.columnName].map(columnData.conversionMap)  # type: ignore
    return convertedData


sexConverter = partial(convertColToNumeric, columnData=SexConversion)
EmbarkedConverter = partial(convertColToNumeric, columnData=EmbarkedConversion)

convertDataToNumeric = compose(sexConverter, EmbarkedConverter)
cleanData = compose(cleanAge, cleanFare, cleanEmbarked, dropIrrelent)

processedData = compose(cleanData, convertDataToNumeric)(data)

processedData

import matplotlib
import matplotlib.pyplot as plt
# %%
import seaborn as sns

matplotlib.use("ipympl")
plt.ioff()

processedDataCorrMatrix = processedData.corr().round(3)
sns.heatmap(processedDataCorrMatrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# %%
def survivalFrame(feature: str, df: DataFrame) -> DataFrame:
    survived = df[df["Survived"] == 1][feature].value_counts()
    dead = df[df["Survived"] == 0][feature].value_counts()
    deathFrame = DataFrame([survived, dead])
    deathFrame.index = ["Survived", "Dead"]
    return deathFrame


survivalFrame("Sex", processedData).rename(columns={1: "Female", 0: "Male"}).plot.bar()
plt.show()

# %%
pclass = survivalFrame("Pclass", processedData).sort_index(axis=1)
pclass.plot.bar()
plt.show()
