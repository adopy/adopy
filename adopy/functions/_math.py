from typing import Union

import numpy as np

__all__ = ['inv_logit', 'logsumexp']


def inv_logit(x: Union[float, np.ndarray]):
    """Calculate the inverse logit value of a given number :math:`x`."""
    return np.divide(1, 1 + np.exp(-x))


def logsumexp(x: np.ndarray, axis: int = 0):
    return np.log(np.sum(np.exp(x), axis=axis))
