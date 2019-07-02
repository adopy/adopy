from collections import OrderedDict
from typing import (
    Any, Callable, Dict, Iterable, Optional, List, Tuple,
    TypeVar
)

import numpy as np
import pandas as pd

data_like = TypeVar(
    'data_like',
    Dict[str, Any],
    'OrderedDict[str, Any]',
    pd.DataFrame
)

integer_like = TypeVar(
    'integer_like',
    int,
    np.int
)

number_like = TypeVar(
    'number_like',
    float,
    int,
    np.float,
    np.int
)

array_like = TypeVar(
    'array_like',
    number_like,
    Iterable[Any],
    np.ndarray
)

vector_like = TypeVar(
    'vector_like',
    Iterable[number_like],
    np.ndarray
)

matrix_like = TypeVar(
    'matrix_like',
    Iterable[Iterable[number_like]],
    np.ndarray
)
