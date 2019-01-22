import numpy as np

__all__ = ['expand_multiple_dims', 'make_vector_shape']


def expand_multiple_dims(x: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Expand the dimensions of a given array.

    >>> x = np.ones(3, 4, 2)
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
