import pytest
from adopy import GridSpace, Task


@pytest.fixture
def grid_design():
    return GridSpace({"x1": [1, 2, 3], "x2": [4, 5, 6]})


@pytest.fixture
def grid_response():
    return GridSpace({"y": [0, 1]})


def test_task(grid_design, grid_response):
    task = Task(
        name="Test",
        designs=["x1", "x2"],
        responses=["y"],
        grid_design=grid_design,
        grid_response=grid_response,
    )

    assert task.name == "Test"

    assert task.designs == ("x1", "x2")

    assert task.responses == ("y",)

    assert task.grid_design is grid_design

    assert task.grid_response is grid_response
