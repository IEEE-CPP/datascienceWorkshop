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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import plot_tree
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
### Pairplot
this is a way of looking at a lot of graphs quickly
"""
# %%
sns.pairplot(processedData, hue="Survived")
plt.show()

# %% [markdown]
"""
### Survival by Sex
"""
# %%
# We don't want to relable our axes
processedDataCategoricalSex = compose(cleanData, embarkedConverter)(data)
sns.countplot(x="Sex", data=processedDataCategoricalSex)
plt.show()
# well that sucks, both sexes are the same color

# %%
# luckily we can fix that with hue
sns.countplot(x="Sex", hue="Sex", data=processedDataCategoricalSex)
plt.show()

# %% [markdown]
"""
hue is actually really powerful. It enables us to break things down by other variables 
"""

# %%
sns.countplot(x="Survived", hue="Sex", data=processedDataCategoricalSex)
plt.show()

# %% [markdown]
"""
### Survival by class
"""
# %%
sns.countplot(x="Survived", hue="Pclass", data=processedDataCategoricalSex)
plt.show()
# %% [markdown]
"""
you can even break it down by multiple things, although it requires some work
"""
# %%
sns.countplot(
    x=processedDataCategoricalSex["Survived"],
    hue=processedDataCategoricalSex[["Pclass", "Sex"]].apply(tuple, axis=1),
)
plt.show()

# %% [markdown]
"""
### Survival by age
"""
# %%
sns.lineplot(x="Survived", y="Age", hue="Sex", data=processedDataCategoricalSex)
plt.show()

# %% [markdown]
"""
That's a horrible way to display the data and doesn't really tell us anything
"""
# %%
sns.boxplot(x="Survived", y="Age", hue="Sex", data=processedDataCategoricalSex)
plt.show()

# %% [markdown]
"""
you can even overlay plots, admittedy thisn't isn't the best representation...
"""

# %%
sns.boxplot(x="Survived", y="Age", hue="Sex", data=processedDataCategoricalSex)
sns.stripplot(x="Survived", y="Age", hue="Sex", data=processedDataCategoricalSex)
plt.show()

# %% [markdown]
"""
### Survival of women by age
"""
# %% [markdown]
"""
You can create new plots that meet certain conditions by modifying your dataframe
"""
# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Sex"] == "female"]
sns.countplot(x="Age", hue="Survived", data=df)
plt.show()

# %% [markdown]
"""
That was unreadable, lets go back to the histogram
"""

# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Sex"] == "female"]
sns.histplot(x="Age", hue="Survived", data=df)
plt.show()

# %% [markdown]
"""
### Survival of men by age
"""

# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Sex"] == "male"]
sns.histplot(x="Age", hue="Survived", data=df)
plt.show()


# %% [markdown]
"""
### Survival of children by age
"""

# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Age"] < 18]
sns.histplot(x=df["Age"], hue=df["Survived"])
plt.show()

# %% [markdown]
"""
The data isn't really displayed quite how we want it.  Things seem to be combined weirdly.  lets try a count plot
"""

# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Age"] < 18]
sns.countplot(
    x=df["Age"],
    hue=df["Survived"],
)
plt.show()

# %% [markdown]
"""
No thats not right either, lets try modifying the histogram
"""

# %%
df = processedDataCategoricalSex[processedDataCategoricalSex["Age"] < 18]
sns.histplot(x=df["Age"], hue=df["Survived"], discrete=True)
plt.show()

# %% [markdown]
"""
The problem with looking at the children is that there simply isn't a lot of data.
We can still calculate what your chance of surviving as a child is compared to adults
"""
# %%
childDf = processedDataCategoricalSex[processedDataCategoricalSex["Age"] < 18]
adultDf = processedDataCategoricalSex[processedDataCategoricalSex["Age"] >= 18]

boyDf = childDf[childDf["Sex"] == "male"]
girlDf = childDf[childDf["Sex"] == "female"]
manDf = adultDf[adultDf["Sex"] == "male"]
womanDf = adultDf[adultDf["Sex"] == "female"]
survivalPercent = lambda df: df["Survived"].value_counts(normalize=True) * 100
survivalDict = {
    "childSurvive": survivalPercent(childDf),
    "adultSurvive": survivalPercent(adultDf),
    "boySurvive": survivalPercent(boyDf),
    "girlSurvive": survivalPercent(girlDf),
    "manSurvive": survivalPercent(manDf),
    "womanSurvive": survivalPercent(womanDf),
}
survivalDict

# %%
data

# %%
processedData

# %% [markdown]
"""
# Machine learning
"""

# %% [markdown]
"""
### Random Forest
"""

# %%
X = processedData.drop(["Survived"], axis=1)
y = processedData["Survived"]
randomState = 42

# splitting up the dataset for teesting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=randomState
)
randomForest = RandomForestClassifier(n_estimators=100, random_state=randomState)
randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

# %% [markdown]
"""
### Fine Tuning
"""

# %%
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
# %% [markdown]
"""
### visualize the model
"""
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
        "Age": [42],
        "SibSp": [1],
        "Parch": [0],
        "Fare": [30],
        "Embarked": [2],
    }
)
testPassengerPrediction = randomForest.predict(testPassenger)
print(testPassengerPrediction)
