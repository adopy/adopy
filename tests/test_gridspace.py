from jax import numpy as jnp
from adopy import GridSpace


def test_space_empty():
    gs = GridSpace({})

    assert len(gs) == 0, "The number of grid points should be 0."

    assert len(gs.labels) == 0, "The number of labels should be 0."

    assert jnp.isnan(gs.value), "The grid array should be nan."


def test_space_1d():
    gs = GridSpace({"x1": [1, 2, 3, 4, 5]})

    assert len(gs) == 5, "The number of grid points should be 5."

    assert len(gs.labels) == 1, "The number of labels should be 1."

    assert gs.value.shape == (5, 1), "The grid array should be shaped as (5, 1)."


def test_space_2d():
    gs = GridSpace({"x1": [1, 2, 3, 4], "x2": [1, 2, 3, 4, 5, 6]})

    assert len(gs) == 24, "The number of grid points should be 24."

    assert len(gs.labels) == 2, "The number of labels should be 2."

    assert gs.value.shape == (24, 2), "The grid array should be shaped as (24, 2)."


def test_space_2d_joint():
    gs = GridSpace(
        {
            ("x1", "x2"): [
                (x1, x2) for x1 in [1, 2, 3, 4, 5] for x2 in [1, 2, 3, 4, 5] if x1 < x2
            ]
        }
    )

    assert len(gs) == 10, "The number of grid points should be 10."

    assert len(gs.labels) == 2, "The number of labels should be 2."

    assert gs.value.shape == (10, 2), "The grid array should be shaped as (10, 2)."
