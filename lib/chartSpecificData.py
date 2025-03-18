import matplotlib as plt
from pandas import DataFrame


def survivalFrame(feature: str, df: DataFrame) -> DataFrame:
    # grabbing all the people who survived with respect to a specific feature
    survived = df[df["Survived"] == 1][feature].value_counts()
    dead = df[df["Survived"] == 0][feature].value_counts()
    deathFrame = DataFrame([survived, dead])
    deathFrame.index = ["Survived", "Dead"]
    return deathFrame
