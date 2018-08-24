from __future__ import absolute_import, division, print_function

import numpy as np

EPS = np.finfo(np.float).eps


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


def log_lik_bern(y, p):
    return y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)


def marginalize(post, grid_param, axis):
    mp = {}
    for value, p in zip(grid_param[:, axis], post):
        k = value if np.isscalar(value) else tuple(value)
        mp[k] = mp.get(k, 0) + p
    return mp


def sample_from_prob_dict(mp, n):
    points = list(mp.keys())
    counts = np.array(list(mp.values())) * n
    counts = np.round(counts).astype(np.int)
    return np.repeat(points, counts, axis=0)


def expand_multiple_dims(x, pre, post):
    ret = x
    for _ in range(pre):
        ret = np.expand_dims(ret, 0)
    for _ in range(post):
        ret = np.expand_dims(ret, -1)
    return ret


def get_nearest_grid_index(design, designs):
    dims_designs = designs.shape[:-1]
    dim = len(dims_designs)
    d = design.reshape(make_vector_shape(dim + 1, dim))
    idx = np.argmin(np.power(designs - d, 2).sum(-1))
    return np.unravel_index(idx, dims_designs)


def get_random_design_index(designs):
    dims_designs = designs.shape[:-1]
    num_possible_designs = np.int(np.prod(designs.shape[:-1]))
    idx = np.random.randint(0, num_possible_designs - 1)
    return np.unravel_index(idx, dims_designs)


def make_vector_shape(n, axis=0):
    ret = np.ones(n)
    ret[axis] = -1
    return ret.astype(np.int)


def make_grid_matrix(*args):
    assert all([isinstance(x, np.ndarray) for x in args])
    assert all([len(np.shape(x)) in {1, 2} for x in args])

    n_dims = len(args)

    n_d_each = [1 if len(np.shape(x)) == 1 else np.shape(x)[1] for x in args]
    n_d_prev = np.cumsum(n_d_each) - n_d_each
    n_d_total = sum(n_d_each)

    grids = []
    for i, g in enumerate(args):
        dim_grid = np.append(make_vector_shape(n_dims, i), n_d_total)

        g_2d = g.reshape(-1, 1) if n_d_each[i] == 1 else g
        grid = np.pad(g_2d, [(0, 0), (n_d_prev[i], n_d_total - n_d_prev[i] - n_d_each[i])],
                      str('constant')).reshape(dim_grid)
        grids.append(grid)

    grid_mat = sum(grids).reshape(-1, n_d_total)

    return grid_mat
