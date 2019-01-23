def const_positive(x):
    """A constraint function that :math:`x > 0`."""
    return x > 0


def const_01(x):
    """A constraint function that :math:`0 < x < 1`."""
    return 0 < x < 1
