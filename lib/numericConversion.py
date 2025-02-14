from dataclasses import dataclass
from functools import partial
from typing import Callable

from pandas import DataFrame


@dataclass
class NumericConversionData:
    columnName: str
    conversionMap: dict


def convertColToNumeric(
    data: DataFrame, columnData: NumericConversionData
) -> DataFrame:
    convertedData = data.copy()
    convertedData[columnData.columnName] = data[columnData.columnName].map(
        columnData.conversionMap
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
