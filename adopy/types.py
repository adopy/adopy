from collections import OrderedDict
from typing import (
    Any, Callable, Dict, Iterable, Optional, List, Tuple,
    TypeVar
)

import numpy as np
import pandas as pd

TYPE_DATA = TypeVar(
    'DATA',
    Dict[str, Any],
    'OrderedDict[str, Any]',
    pd.DataFrame
)

TYPE_NUMBER = TypeVar(
    'Number_like',
    float,
    int,
    np.float,
    np.int
)

TYPE_ARRAY = TypeVar(
    'array_like',
    TYPE_NUMBER,
    Iterable[Any],
    np.ndarray
)

TYPE_VECTOR = TypeVar(
    'vector_like',
    Iterable[TYPE_NUMBER],
    np.ndarray
)

TYPE_MATRIX = TypeVar(
    'matrix_like',
    Iterable[Iterable[TYPE_NUMBER]],
    np.ndarray
)
