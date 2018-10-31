from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ['inv_logit']


def inv_logit(x):
    """Calculate the inverse logit value of given number.

    Parameters
    ----------
    x : float or array_like
        Value as a logit

    Returns
    -------
    float or array_like
        Inverse logit of the given value.

    """
    return np.divide(1, 1 + np.exp(-x))
