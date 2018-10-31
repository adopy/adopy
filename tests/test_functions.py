from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy.functions import inv_logit
from adopy.functions import marginalize, expand_multiple_dims


def test_inv_logit():
    assert inv_logit(-np.inf) == 0.0
    assert inv_logit(0) == 0.5
    assert inv_logit(np.inf) == 1.0


def test_expand_multiple_dims():
    x = np.arange(4).reshape(-1)

    assert expand_multiple_dims(x, 1, 0).shape == (1, 4)
    assert expand_multiple_dims(x, 0, 2).shape == (4, 1, 1)

    y = np.arange(12).reshape(3, 4)

    assert expand_multiple_dims(y, 3, 2).shape == (1, 1, 1, 3, 4, 1, 1)


def test_get_nearest_design_index():
    pass


def test_get_random_design_index():
    pass


def test_make_vector_shape():
    pass


def test_make_grid_design():
    pass


if __name__ == '__main__':
    pytest.main()
