"""
Define functions for constraints on model parameters.

These constraints should be defined as named functions, since unnamed functions
like lambda cannot be serialized by pickle.
"""


def const_positive(x):
    return x > 0


def const_01(x):
    return 0 < x < 1
