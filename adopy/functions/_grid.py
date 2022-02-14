from typing import Iterable, Tuple, TypeVar

import numpy as np

from adopy.types import array_like, vector_like
from jax import numpy as jnp

__all__ = [
    "marginalize",
    "get_nearest_grid_index",
]

GK = TypeVar("GK", str, Tuple[str])
GV = TypeVar("GV", Iterable, np.ndarray, jnp.ndarray)


def marginalize(post, grid_param, axis):
    """Return marginal distributions from grid-shaped posteriors"""
    mp = {}
    for value, p in zip(np.array(grid_param)[:, axis], post):
        k = value if np.isscalar(value) else tuple(value)
        mp[k] = mp.get(k, 0) + p
    return mp


def get_nearest_grid_index(design: vector_like, designs: array_like) -> int:
    """
    Find the index of the best matching row vector to the given vector.
    """
    ds = designs
    d = design.reshape(1, -1)
    return np.square(ds - d).sum(-1).argsort()[0]
