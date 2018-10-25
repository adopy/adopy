from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ['log_lik_bern']

EPS = np.finfo(np.float).eps


def log_lik_bern(y, p):
    """Log likelihood for Bernoulli random variable"""
    return y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)
