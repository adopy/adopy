from typing import Dict, Iterable, List, Tuple, TypeVar, Optional, Any

import numpy as np
import pandas as pd
from jax import numpy as jnp

from adopy.functions import make_vector_shape

__all__ = ["GridSpace"]


K = TypeVar("K", str, Tuple[str])
V = TypeVar("V", Iterable, np.ndarray, jnp.ndarray)


class GridSpace(object):
    def __init__(
        self,
        axes: Dict[K, V],
        order: Optional[List[str]] = None,
        dtype: Optional[Any] = jnp.float32,
    ):
        assert isinstance(axes, dict), "axes argument should be a dictionary."

        assert all(
            isinstance(k, str)  # a string scalar
            or (
                isinstance(k, tuple) and all(isinstance(v, str) for v in k)
            )  # a string tuple
            for k in axes.keys()
        ), "Keys of axes argument should be a string or a tuple of strings."

        assert all(
            len(jnp.shape(v)) in {1, 2} for v in axes.values()
        ), "Values of axes argument should be a 1d or 2d array."

        self._value, self._names = self.process_axes(axes, dtype)
        self._order = order if order else self._names
        self._dtype = dtype

    @property
    def value(self):
        return self._value

    @property
    def names(self):
        return self._names

    @property
    def order(self):
        return self._order

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._value = self._value.astype(value)

    def __repr__(self):
        return f"GridSpace(names={self.names})"

    def to_dataframe(self):
        return pd.DataFrame(self.value, columns=self.names)[self.order]

    @staticmethod
    def process_axes(axes, dtype) -> Tuple[jnp.array, List[str]]:
        n_dims = len(axes)

        if not n_dims:
            return jnp.array(None), []

        axes_names = list(axes.keys())
        axes_values = [jnp.array(x) for x in axes.values()]

        n_d_each = jnp.array(
            [1 if len(x.shape) == 1 else x.shape[1] for x in axes_values]
        )
        n_d_prev = jnp.cumsum(n_d_each) - n_d_each
        n_d_total = sum(n_d_each)

        labels = []  # type: List[str]
        grids = []  # type: List[jnp.ndarray]

        for i, (k, g) in enumerate(zip(axes_names, axes_values)):
            dim_grid = jnp.append(make_vector_shape(n_dims, i), n_d_total)

            if isinstance(k, str):
                labels.append(k)
            else:
                labels.extend(k)

            # Make a grid as a 2d matrix
            g_2d = g.reshape((-1, 1)) if n_d_each[i] == 1 else g

            # Convert to a given dtype
            g_2d = g_2d.astype(dtype)

            grid = jnp.pad(
                g_2d,
                [(0, 0), (n_d_prev[i], n_d_total - n_d_prev[i] - n_d_each[i])],
                "constant",
            ).reshape(dim_grid)

            grids.append(grid)

        grid_mat = sum(grids, jnp.zeros_like(grids[0])).reshape(-1, n_d_total)

        return grid_mat, labels
