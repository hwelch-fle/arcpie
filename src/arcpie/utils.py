"""Module for internal utility functions to share between modules"""

import numpy as np
from typing import Any
from datetime import datetime
import builtins

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