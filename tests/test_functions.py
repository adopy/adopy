import numpy as np
import pytest

from adopy.functions import (
    expand_multiple_dims, get_nearest_grid_index, make_vector_shape,
    marginalize,
)


def test_get_nearest_grid_index():
    X = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 1],
        [3, 4, 1, 2],
        [4, 1, 2, 3],
    ])

    x0 = X[0].reshape(1, -1)
    x1 = X[1].reshape(1, -1)
    x2 = X[2].reshape(1, -1)
    x3 = X[3].reshape(1, -1)

    assert get_nearest_grid_index(x0, X) == 0
    assert get_nearest_grid_index(x1, X) == 1
    assert get_nearest_grid_index(x2, X) == 2
    assert get_nearest_grid_index(x3, X) == 3


def test_expand_multiple_dims():
    x = np.arange(4).reshape(-1)

    assert expand_multiple_dims(x, 1, 0).shape == (1, 4)
    assert expand_multiple_dims(x, 0, 2).shape == (4, 1, 1)

    y = np.arange(12).reshape(3, 4)

    assert expand_multiple_dims(y, 3, 2).shape == (1, 1, 1, 3, 4, 1, 1)


def test_make_vector_shape_uses_integer_dtype():
    shape = make_vector_shape(3, 1)

    assert shape.tolist() == [1, -1, 1]
    assert np.issubdtype(shape.dtype, np.integer)

if __name__ == '__main__':
    pytest.main()
