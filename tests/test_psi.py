import numpy as np
import pytest

from adopy.tasks.psi import ModelLogistic, ModelWeibull, ModelProbit, EnginePsi


@pytest.fixture()
def grid_design():
    stimulus = np.linspace(20 * np.log10(.05), 20 * np.log10(400), 20)
    designs = dict(stimulus=stimulus)
    return designs


@pytest.fixture()
def grid_param():
    guess_rate = [0.5]
    lapse_rate = [0.05]
    threshold = np.linspace(20 * np.log10(.1), 20 * np.log10(200), 20)
    slope = np.linspace(0, 10, 11)[1:]
    params = dict(guess_rate=guess_rate, lapse_rate=lapse_rate,
                  threshold=threshold, slope=slope)
    return params


@pytest.mark.parametrize('design_type', ['optimal', 'staircase', 'random'])
@pytest.mark.parametrize('model', [ModelLogistic, ModelWeibull, ModelProbit])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, grid_design, grid_param, response):
    psi = EnginePsi(model=model(),
                    grid_design=grid_design, grid_param=grid_param)
    d = psi.get_design(design_type)
    psi.update(d, response)


if __name__ == '__main__':
    pytest.main()
