import numpy as np
import pandas as pd
from pandas import DataFrame


def addFamilyCountData(data: DataFrame):
    df = data.copy()
    # SibSp is number of siblings and spouses
    # Parch is number of parents/children
    df["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] > 1, 0, 1)
    return df


def addIsAdult(data: DataFrame) -> DataFrame:
    df = data.copy()
    df["isAdult"] = pd.cut(df["Age"], [0, 15, 100], labels=[0, 1])
    return df
