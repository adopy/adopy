import pytest
import jax
from jax import numpy as jnp, scipy as jsp
from adopy import GridSpace, Task, Model, Engine


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


@pytest.fixture
def model(task, grid_param):
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

    return TestModel()


def test_engine(task, model):
    engine = Engine(task=task, model=model)
    design = engine.get_design()

    assert isinstance(design, dict)
    assert tuple(design.keys()) == task.designs

    post_mean = engine.post_mean
    post_sd = engine.post_sd

    assert isinstance(post_mean, dict)
    assert isinstance(post_sd, dict)

    assert tuple(post_mean.keys()) == model.params
    assert tuple(post_sd.keys()) == model.params

    response = {"y": 1}
    engine.update(design, response)
