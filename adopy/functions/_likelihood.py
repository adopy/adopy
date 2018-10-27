from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ['log_lik_bernoulli', 'log_lik_categorical']

EPS = np.finfo(np.float).eps


def log_lik_bernoulli(y, p):
    """Log likelihood for Bernoulli random variable"""
    return y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)


def log_lik_categorical(ys, ps):
    ret = 0.
    for y, p in zip(ys, ps):
        ret += y * np.log(p + EPS)
    return ret
