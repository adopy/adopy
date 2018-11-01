from __future__ import absolute_import, division, print_function

from typing import Dict, Iterable, List, Tuple, TypeVar

import numpy as np
import pandas as pd

from ._utils import make_vector_shape

__all__ = [
    'marginalize', 'get_nearest_grid_index', 'get_random_design_index', 'make_grid_matrix'
]

GK = TypeVar('GK', str, Tuple[str])
GV = TypeVar('GV', Iterable, np.ndarray)


def marginalize(post, grid_param, axis):
    mp = {}
    for value, p in zip(grid_param[:, axis], post):
        k = value if np.isscalar(value) else tuple(value)
        mp[k] = mp.get(k, 0) + p
    return mp


def get_nearest_grid_index(design, designs):
    # type: (pd.Series, pd.DataFrame) -> int
    return int(np.argmin(np.square((designs - design).values).sum(-1)))


def get_random_design_index(designs):
    dims_designs = designs.shape[:-1]
    num_possible_designs = np.int(np.prod(designs.shape[:-1]))
    idx = np.random.randint(0, num_possible_designs - 1)
    return np.unravel_index(idx, dims_designs)


def make_grid_matrix(axes_dict):
    # type: (Dict[GK, GV]) -> pd.DataFrame
    assert isinstance(axes_dict, dict)
    assert all([len(np.shape(x)) in {1, 2} for x in axes_dict.values()])

    n_dims = len(axes_dict)

    n_d_each = [1 if len(np.shape(x)) == 1 else np.shape(x)[1] for x in axes_dict.values()]
    n_d_prev = np.cumsum(n_d_each) - n_d_each
    n_d_total = sum(n_d_each)

    columns = []  # type: List[str]
    grids = []  # type: List[np.ndarray]
    for i, (k, g) in enumerate(axes_dict.items()):
        dim_grid = np.append(make_vector_shape(n_dims, i), n_d_total)

        if isinstance(k, str):
            columns.append(k)
        else:
            columns.extend(k)
        g_2d = np.reshape(g, (-1, 1)) if n_d_each[i] == 1 else g
        grid = np.pad(g_2d, [(0, 0), (n_d_prev[i], n_d_total - n_d_prev[i] - n_d_each[i])],
                      'constant').reshape(dim_grid)
        grids.append(grid)

    grid_mat = np.sum(grids).reshape(-1, n_d_total)

    return pd.DataFrame(grid_mat, columns=columns)
