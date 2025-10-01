"""Module for internal utility functions to share between modules"""

import numpy as np
from typing import Any
from datetime import datetime
import builtins

from .featureclass import (
    FeatureClass, 
    Table,
    GeometryType,
    where,
    count,
)

from .database import (
    Dataset,
)

def cast_type(dtype: np.dtype[Any]) -> type:
    match dtype.type:
        case np.int_:
            return int
        case np.float_:
            return float
        case np.str_:
            return str
        case np.datetime64:
            return datetime
        case builtins.object:
            return builtins.object
        case _:
            return builtins.object

def convert_dtypes(dtypes: np.dtype[Any]) -> dict[str, type]:
    return {field: cast_type(dtypes[field]) for field in dtypes.names or {}}

def get_subtype_count(fc: Table, drop_empty: bool=False) -> dict[str, int]:
    return {
        subtype['Name']: cnt
        for code, subtype in fc.subtypes.items() 
        if fc.subtype_field # has Subtypes
        and (
            (cnt := count(fc[where(f'{fc.subtype_field} = {code}')])) # Get count
            or drop_empty # Drop Empty counts?
        )
    }

def get_subtype_counts(gdb: Dataset, drop_empty: bool=False) -> dict[str, dict[str, int]]:
    feats: list[Table] = [*gdb.feature_classes.values(), *gdb.tables.values()]
    return {
        fc.name: counts
        for fc in feats
        if (counts := get_subtype_count(fc))
        or not drop_empty
    }