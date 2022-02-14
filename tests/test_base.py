from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit as inv_logit
from scipy.stats import bernoulli

from adopy import Engine, Model, Task


@pytest.fixture()
def task():
    return Task(
        name="Psychometric function estimation",
        designs=["stimulus"],
        responses=["choice"],
    )


@pytest.fixture()
def task_noname():
    return Task(designs=["stimulus"], responses=["choice"])


def test_task(task, task_noname):
    # task.name
    assert task.name == "Psychometric function estimation"
    assert task_noname.name is None

    # task.designs
    assert task.designs == ["stimulus"]

    # task.responses
    assert task.responses == ["choice"]

    # repr(task)
    assert (
        repr(task)
        == "Task('Psychometric function estimation', designs=['stimulus'], responses=['choice'])"
    )


def func_logistic_log_lik(choice, stimulus, guess_rate, lapse_rate, threshold, slope):
    f = inv_logit(slope * (stimulus - threshold))
    p = guess_rate + (1 - guess_rate - lapse_rate) * f
    return bernoulli.logpmf(choice, p)


@pytest.fixture()
def model(task):
    return Model(
        name="Logistic",
        task=task,
        params=["guess_rate", "lapse_rate", "threshold", "slope"],
        func=func_logistic_log_lik,
    )


@pytest.fixture()
def model_noname(task):
    return Model(
        task=task,
        params=["guess_rate", "lapse_rate", "threshold", "slope"],
        func=func_logistic_log_lik,
    )


def test_model(model, model_noname, task):
    # model.name
    assert model.name == "Logistic"
    assert model_noname.name is None

    # model.task
    assert model.task is task
    assert model_noname.task is task

    # model.params
    assert model.params == ["guess_rate", "lapse_rate", "threshold", "slope"]

    # model.compute()
    assert model.compute(
        choice=1, stimulus=10, guess_rate=0.5, lapse_rate=0.05, threshold=8, slope=2
    ) == func_logistic_log_lik(
        choice=1, stimulus=10, guess_rate=0.5, lapse_rate=0.05, threshold=8, slope=2
    )
    assert model.compute(1, 10, 0.5, 0.05, 8, 2) == func_logistic_log_lik(
        1, 10, 0.5, 0.05, 8, 2
    )

    # repr(task)
    assert (
        repr(model)
        == "Model('Logistic', params=['guess_rate', 'lapse_rate', 'threshold', 'slope'])"
    )
    assert (
        repr(model_noname)
        == "Model(params=['guess_rate', 'lapse_rate', 'threshold', 'slope'])"
    )


@pytest.fixture()
def grid_design():
    return {"stimulus": np.linspace(20 * np.log10(0.05), 20 * np.log10(400), 120)}


@pytest.fixture()
def grid_param():
    return {
        "threshold": np.linspace(20 * np.log10(0.1), 20 * np.log10(200), 100),
        "slope": np.linspace(0, 10, 101)[0:],
        "guess_rate": [0.5],
        "lapse_rate": [0.05],
    }


@pytest.fixture()
def grid_response():
    return {"choice": [0, 1]}


@pytest.fixture()
def engine(task, model, grid_design, grid_param, grid_response):
    return Engine(
        task=task,
        model=model,
        grid_design=grid_design,
        grid_param=grid_param,
        grid_response=grid_response,
    )


def test_engine(engine, task, model):
    # engine.task
    assert engine.task is task

    # engine.model
    assert engine.model is model

    # engine.post_mean
    assert isinstance(engine.post_mean, pd.Series)
    assert len(engine.post_mean) == 4

    # engine.post_cov
    assert isinstance(engine.post_cov, np.ndarray)
    assert np.shape(engine.post_cov) == (4, 4)

    # engine.post_sd
    assert isinstance(engine.post_sd, pd.Series)
    assert len(engine.post_sd) == 4

    # engine.reset()
    engine.reset()


@pytest.mark.parametrize("design_type", ["optimal", "random"])
def test_engine_get_design(engine, design_type):
    _ = engine.get_design(design_type)


@pytest.mark.parametrize("choice", [0, 1])
def test_engine_single_update(engine, choice):
    design = engine.get_design()
    response = {"choice": choice}

    try:
        engine.update(design, response)
    except Exception:
        pytest.fail("Updating with single observation failed.")


def test_engine_multiple_update(engine):
    try:
        engine.update([engine.get_design("random") for _ in range(3)], {"choice": 1})

        pytest.fail(
            "Failed to detect whether both design and response are " "given as lists."
        )
    except Exception:
        pass

    try:
        engine.update(engine.get_design("random"), [{"choice": 1} for _ in range(3)])

        pytest.fail(
            "Failed to detect whether both design and response are " "given as lists."
        )
    except Exception:
        pass

    try:
        engine.update(
            [engine.get_design("random") for _ in range(3)],
            [{"choice": 1} for _ in range(3)],
        )

    except Exception:
        pytest.fail("Failed to update multiple observations.")


if __name__ == "__main__":
    pytest.main(__file__)
