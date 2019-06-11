from typing import Any, Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd

from adopy.types import data_like

__all__ = [
    'extract_vars_from_data',
    'expand_multiple_dims',
    'make_vector_shape',
]


def extract_vars_from_data(data: data_like,
                           keys: Iterable[str]) -> 'OrderedDict[str, Any]':
    """
    Extract variables corresponding to given keys from the data. The data can
    be a dictionary, an ``OrderedDict``, or a pandas.DataFrame.

    Examples
    --------
    >>> data = {'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]}
    >>> extract_vars_from_data(data, ['x', 'y'])
    OrderedDict([('x', [1, 2, 3]), ('y', [4, 5, 6])])
    >>> extract_vars_from_data(data, ['a'])
    Traceback (most recent call last):
        ...
    RuntimeError: key 'a' is not available.
    """
    ret = OrderedDict()  # type: OrderedDict[str, Any]
    for k in keys:
        if k not in data:
            raise RuntimeError("key '{}' is not available.".format(k))
        if isinstance(data, pd.DataFrame):
            ret[k] = data[k].values
        else:
            ret[k] = data[k]
    return ret


def expand_multiple_dims(x: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Expand the dimensions of a given array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones((3, 4, 2))
    >>> print(x.shape)
    (3, 4, 2)
    >>> y = expand_multiple_dims(x, 2, 3)
    >>> print(y.shape)
    (1, 1, 3, 4, 2, 1, 1, 1)
    """
    ret = x
    for _ in range(pre):
        ret = np.expand_dims(ret, 0)
    for _ in range(post):
        ret = np.expand_dims(ret, -1)
    return ret


def make_vector_shape(n: int, axis: int = 0) -> np.ndarray:
    ret = np.ones(n)
    ret[axis] = -1
    return ret.astype(np.int)
