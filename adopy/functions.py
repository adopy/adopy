from typing import Dict, Tuple, TypeVar, List, Union

from jax import numpy as jnp

__all__ = [
    "marginalize",
]

MK = TypeVar("MK", float, Tuple[float])


def marginalize(
    post: jnp.ndarray, grid: jnp.ndarray, axis: Union[int, List[int]]
) -> Dict[MK, float]:
    """Return marginal distributions from grid-shaped posteriors"""
    assert len(post) == len(grid)

    ret = {}
    for v, p in zip(grid[:, axis], post):
        vv = v.tolist()
        k = vv if jnp.isscalar(vv) else tuple(vv)
        ret[k] = ret.get(k, 0) + p.tolist()

    return dict(sorted(ret.items(), key=lambda x: x[0]))
