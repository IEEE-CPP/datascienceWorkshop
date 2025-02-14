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
