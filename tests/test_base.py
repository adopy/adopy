from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy import Task, Model, Engine
from adopy.functions import inv_logit


@pytest.fixture()
def task():
    return Task(name='Psi', designs=['stimulus'], responses=[0, 1])


@pytest.fixture()
def model(task):
    def func_logistic(stimulus, guess_rate, lapse_rate, threshold, slope):
        return guess_rate + (1 - guess_rate - lapse_rate) \
            * inv_logit(slope * (stimulus - threshold))

    return Model(
        name='Logistic',
        task=task,
        params=['guess_rate', 'lapse_rate', 'threshold', 'slope'],
        func=func_logistic)


@pytest.fixture()
def designs():
    stimulus = np.linspace(20 * np.log10(.05), 20 * np.log10(400), 120)
    designs = dict(stimulus=stimulus)
    return designs


@pytest.fixture()
def params():
    guess_rate = [0.5]
    lapse_rate = [0.05]
    threshold = np.linspace(20 * np.log10(.1), 20 * np.log10(200), 100)
    slope = np.linspace(0, 10, 100)[1:]
    params = dict(guess_rate=guess_rate, lapse_rate=lapse_rate,
                  threshold=threshold, slope=slope)
    return params


@pytest.fixture()
def engine(task, model, designs, params):
    return Engine(task=task, model=model, designs=designs, params=params)


@pytest.mark.parametrize('design_type', ['optimal', 'random'])
@pytest.mark.parametrize('y_obs', [0, 1])
def test_overall_workflow(engine, design_type, y_obs):
    d_opt = engine.get_design(design_type)
    engine.update(d_opt, y_obs)


if __name__ == '__main__':
    pytest.main()
