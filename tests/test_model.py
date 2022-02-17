import pytest
import jax
from jax import numpy as jnp, scipy as jsp
from adopy import GridSpace, Task, Model


@pytest.fixture
def grid_design():
    return GridSpace({"x1": [1, 2, 3], "x2": [4, 5, 6]})


@pytest.fixture
def grid_response():
    return GridSpace({"y": [0, 1]})


@pytest.fixture
def grid_param():
    return GridSpace(
        {
            "p0": [-2, -1, 0, 1, 2],
            "p1": [-2, -1, 0, 1, 2],
            "p2": [-2, -1, 0, 1, 2],
        }
    )


@pytest.fixture
def task(grid_design, grid_response):
    return Task(
        name="Test task",
        designs=["x1", "x2"],
        responses=["y"],
        grid_design=grid_design,
        grid_response=grid_response,
    )


def test_model(task, grid_param):
    class TestModel(Model):
        def __init__(self):
            super().__init__(
                name="Test model",
                task=task,
                params=["p0", "p1", "p2"],
                grid_param=grid_param,
            )

        @staticmethod
        @jax.jit
        def compute(x1, x2, y, p0, p1, p2):
            z = p0 + p1 * x1 + p2 * x2
            p = 1.0 / (1.0 + jnp.exp(-z))
            return jsp.stats.bernoulli.logpmf(y, p)

    model = TestModel()

    assert model.name == "Test model"

    assert model.task is task

    assert model.params == ("p0", "p1", "p2")

    assert model.compute(0, 0, 1, 0, 0, 0) == jnp.log(0.5)


def test_model_noname(task, grid_param):
    class TestModel(Model):
        def __init__(self):
            super().__init__(
                task=task,
                params=["p0", "p1", "p2"],
                grid_param=grid_param,
            )

        @staticmethod
        @jax.jit
        def compute(x1, x2, y, p0, p1, p2):
            z = p0 + p1 * x1 + p2 * x2
            p = 1.0 / (1.0 + jnp.exp(-z))
            return jsp.stats.bernoulli.logpmf(y, p)

    model = TestModel()

    assert model.name is None

    assert model.task is task

    assert model.params == ("p0", "p1", "p2")

    assert model.compute(0, 0, 1, 0, 0, 0) == jnp.log(0.5)
