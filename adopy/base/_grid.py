from typing import Dict, Iterable, List, Tuple, TypeVar, Optional, Any

import numpy as np
import pandas as pd
from jax import numpy as jnp

__all__ = ["GridSpace"]


K = TypeVar("K", str, Tuple[str, ...])
V = TypeVar("V", Iterable, np.ndarray, jnp.ndarray)


def make_vector_shape(n: int, axis: int = 0) -> jnp.ndarray:
    return jnp.ones(n, dtype=int).at[axis].set(-1)


def process_axes(grid, dtype=jnp.float32) -> Tuple[jnp.array, List[str]]:
    n_dims = len(grid)

    if not n_dims:
        return jnp.array(None), []

    axes_names = list(grid.keys())
    axes_values = [jnp.array(x) for x in grid.values()]

    n_d_each = jnp.array([1 if len(x.shape) == 1 else x.shape[1] for x in axes_values])
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

    # Sort axes alphabetically to their labels
    idx_sort = np.argsort(labels)

    return grid_mat[:, idx_sort], sorted(labels)


class GridSpace(object):
    """
    A class for grid definition on a certain space.

    Parameters
    ----------
    grid_def : Dict[Union[str, Tuple[str]], Union[Iterable[float], Iterable[Iterable[float]], np.ndarray, jnp.ndarray]]
        Grid definition as a dictionary object.
    dtype
        Data type for the grid space. Default to :code:`jnp.float32`.
    """

    __name__ = "GridSpace"

    def __init__(
        self,
        grid_def: Dict[K, V],
        dtype: Optional[Any] = jnp.float32,
    ):
        assert isinstance(grid_def, dict), "grid_def argument should be a dictionary."

        assert all(
            isinstance(k, str)  # a string scalar
            or (
                isinstance(k, tuple) and all(isinstance(v, str) for v in k)
            )  # a string tuple
            for k in grid_def.keys()
        ), "Keys of grid_def argument should be a string or a tuple of strings."

        assert all(
            len(jnp.shape(v)) in {1, 2} for v in grid_def.values()
        ), "Values of grid_def argument should be a 1d or 2d array."

        self._value, self._labels = process_axes(grid_def, dtype)
        self._dtype = dtype

    @property
    def value(self):
        return self._value

    @property
    def labels(self):
        return self._labels

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._value = self._value.astype(value)

    def __len__(self):
        try:
            return len(self._value)
        except TypeError:
            return 0

    def __getitem__(self, key) -> Dict[str, float]:
        # return self.value[key]
        return dict(zip(self.labels, self.value[key].tolist()))

    def __repr__(self):
        return f"{self.__name__}(labels={self.labels}, dtype={self.dtype})"

    def astype(self, dtype):
        """Change data type with the given type and return itself"""
        self.dtype = dtype
        return self

    def get_nearest_point(self, x) -> Tuple[int, jnp.ndarray]:
        """
        Find the best matching point in the grid space to the given vector.
        It returns a tuple of the index in the grid space and actual vector.

        Returns
        -------
        point_tuple : Tuple[int, jnp.ndarray]
            2-tuple of the index of the nearest point in GridSpace and its
            actual point
        """
        assert len(jnp.shape(x)) == 1, "The point should be a 1d vector."
        assert jnp.shape(x)[0] == len(self.labels), "The "

        idx = jnp.square(self.value - jnp.reshape(x, (1, -1))).sum(-1).argsort()[0]
        return idx, self.value[idx]

    def to_dataframe(self):
        """Convert GridSpace into Pandas DataFrame."""
        return pd.DataFrame(self.value, columns=self.labels)
