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

# %%
# shortened url to import
# https://n9.cl/ieeedatascience2

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

# %% data Preprocessing [markdown]
"""
# Data Preprocessing

Here we are just getting rid of null values and dropping irrelevant data that we don't need
"""

# %% [markdown]
"""
## Required libraries
"""

# %% [markdown]
"""
### Data cleaning
"""

# %%
from functools import partial
from typing import Callable, List

from pandas import DataFrame


def fillColumnNaWithMedian(data: DataFrame, columnName: str) -> DataFrame:
    columnMedian = data[columnName].median()
    return data.fillna(value={columnName: columnMedian})


def fillColumnNaWithMode(data: DataFrame, columnName: str) -> DataFrame:
    columnMode = data[columnName].mode()
    return data.fillna(value={columnName: columnMode})


def dataDropper(data: DataFrame, columns: List[str]):
    return data.drop(columns, axis=1)


cleanAge: Callable[[DataFrame], DataFrame] = partial(
    fillColumnNaWithMedian, columnName="Age"
)

cleanEmbarked: Callable[[DataFrame], DataFrame] = partial(
    fillColumnNaWithMode, columnName="Embarked"
)

cleanFare: Callable[[DataFrame], DataFrame] = partial(
    fillColumnNaWithMedian, columnName="Fare"
)

dropIrrelevant: Callable[[DataFrame], DataFrame] = partial(
    dataDropper, columns=["Name", "Ticket", "Cabin"]
)

# %% [markdown]
"""
### Numeric Conversion
"""
# %%

from dataclasses import dataclass
from functools import partial
from typing import Callable

from pandas import DataFrame


@dataclass
class NumericConversionData:
    columnName: str
    conversionMap: dict[str, int]


def convertColToNumeric(
    data: DataFrame, columnData: NumericConversionData
) -> DataFrame:
    convertedData = data.copy()
    convertedData[columnData.columnName] = data[columnData.columnName].map(
        columnData.conversionMap  # type: ignore
    )
    return convertedData


SexConversion = NumericConversionData("Sex", {"male": 0, "female": 1})
sexConverter: Callable[[DataFrame], DataFrame] = partial(
    convertColToNumeric, columnData=SexConversion
)

EmbarkedConversion = NumericConversionData("Embarked", {"C": 0, "Q": 1, "S": 2})
embarkedConverter: Callable[[DataFrame], DataFrame] = partial(
    convertColToNumeric, columnData=EmbarkedConversion
)
# %%
cleanData = compose(cleanAge, cleanFare, cleanEmbarked, dropIrrelevant)
convertDataToNumeric = compose(sexConverter, embarkedConverter)

processedData = compose(cleanData, convertDataToNumeric)(data)
processedDataRows = len(processedData)

# %% [markdown]
"""
# Data Exploration Part II and Data Visualization
"""

# %%
ProfileReport(processedData, title="Titanic")

# %%
processedCorrMatrix = processedData.corr().round(3)
sns.heatmap(processedCorrMatrix, annot=True, cmap="coolwarm")
plt.show()

# %%
sns.pairplot(processedData, hue="Survived")
plt.show()

# %%
processDataCategoricalSex = compose(cleanData, embarkedConverter)(data)
sns.countplot(x="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Sex", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Sex", hue="Survived", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Survived", hue="Pclass", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Pclass", hue="Survived", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Pclass", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(
    x=processDataCategoricalSex["Survived"],
    hue=processDataCategoricalSex[["Pclass", "Sex"]].apply(tuple, axis=1),
)
plt.show()

# %%
sns.lineplot(x="Survived", y="Age", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.boxplot(x="Survived", y="Age", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.stripplot(x="Survived", y="Age", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.stripplot(x="Survived", y="Age", hue="Sex", data=processDataCategoricalSex)
sns.boxplot(x="Survived", y="Age", hue="Sex", data=processDataCategoricalSex)
plt.show()

# %%
sns.countplot(x="Age", hue="Survived", data=processDataCategoricalSex)
plt.show()

# %%
sns.histplot(x="Age", hue="Survived", data=processDataCategoricalSex)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Sex"] == "female"]
sns.histplot(x="Age", hue="Survived", data=df)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Sex"] == "male"]
sns.histplot(x="Age", hue="Survived", data=df)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Age"] < 18]
sns.histplot(x="Age", hue="Survived", data=df)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Age"] < 18]
sns.countplot(x="Age", hue="Survived", data=df)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Age"] < 18]
sns.countplot(x="Age", hue="Sex", data=df)
plt.show()

# %%
df = processDataCategoricalSex[processDataCategoricalSex["Age"] < 18]
sns.histplot(x="Age", hue="Survived", data=df, discrete=True)
plt.show()

# %%
childDf = processDataCategoricalSex[processDataCategoricalSex["Age"] < 18]
adultDf = processDataCategoricalSex[processDataCategoricalSex["Age"] >= 18]

boyDf = childDf[childDf["Sex"] == "male"]
girlDf = childDf[childDf["Sex"] == "female"]
manDf = adultDf[adultDf["Sex"] == "male"]
womanDf = adultDf[adultDf["Sex"] == "female"]

survivalPercent = lambda df: df["Survived"].value_counts(normalize=True) * 100

survivalDict = {
    "childSurvive": survivalPercent(childDf),
    "adultSurvive": survivalPercent(adultDf),
    "manSurvive": survivalPercent(manDf),
    "womanSurvive": survivalPercent(womanDf),
    "boySurvive": survivalPercent(boyDf),
    "girlSurvive": survivalPercent(girlDf),
}
survivalDict

# %%
X = processedData.drop(["Survived"], axis=1)
y = processedData["Survived"]
randomState = 42

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=randomState
)
randomForest = RandomForestClassifier(n_estimators=100, random_state=randomState)
randomForest.fit(X_train, y_train)

y_predict = randomForest.predict(X_val)
accuracy = accuracy_score(y_val, y_predict)

paramGrid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 6, 8, 10],
    "criterion": ["gini", "entropy"],
}
gridSearch = GridSearchCV(
    estimator=randomForest, param_grid=paramGrid, cv=5, n_jobs=3, scoring="accuracy"
)
gridSearch.fit(X_train, y_train)

bestParams = gridSearch.best_params_

print(f"bestParams: {bestParams}")
# create new model with better parameters
randomForest2 = gridSearch.best_estimator_
# fit the model
randomForest2.fit(X_train, y_train)
y_pred = randomForest2.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

# %%
tree = randomForest.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=True,
    filled=True,
    fontsize=6,
    rounded=True,
)
plt.show()

# %%
testPassenger = pd.DataFrame.from_dict(
    {
        "PassengerId": [1],
        "Pclass": [1],
        "Sex": [1],
        "Age": [21],
        "SibSp": [2],
        "Parch": [2],
        "Fare": [1000000],
        "Embarked": [1],
    }
)
testPassengerPrediction = randomForest.predict(testPassenger)
print(testPassengerPrediction)

# %%
