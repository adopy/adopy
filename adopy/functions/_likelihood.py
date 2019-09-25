import numpy as np

__all__ = ['log_lik_bernoulli', 'log_lik_categorical']


def log_lik_bernoulli(y, p, dtype=np.float32):
    """Log likelihood for a Bernoulli random variable"""
    eps = np.finfo(dtype).eps * 2
    return y * np.log(p + eps, dtype=dtype) + \
        (1 - y) * np.log(1 - p + eps, dtype=dtype)


def log_lik_categorical(ys, ps, dtype=np.float32):
    """Log likelihood for a categorical random variable"""
    eps = np.finfo(dtype).eps * 2
    ret = 0.
    for y, p in zip(ys, ps):
        ret += y * np.log(p + eps, dtype=dtype)
    return ret
