from typing import Union

import numpy as np

__all__ = ['inv_logit']


def inv_logit(x: Union[float, np.ndarray]):
    """Calculate the inverse logit value of a given number :math:`x`."""
    return np.divide(1, 1 + np.exp(-x))
