from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ['expand_multiple_dims', 'make_vector_shape']


def expand_multiple_dims(x, pre, post):
    # type: (np.ndarray, int, int) -> np.ndarray
    ret = x
    for _ in range(pre):
        ret = np.expand_dims(ret, 0)
    for _ in range(post):
        ret = np.expand_dims(ret, -1)
    return ret


def make_vector_shape(n, axis=0):
    # type: (int, int) -> np.ndarray
    ret = np.ones(n)
    ret[axis] = -1
    return ret.astype(np.int)
