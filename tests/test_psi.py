from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy.tasks.psi import ModelLogistic, ModelWeibull, ModelNormal, EnginePsi


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
    params = dict(guess_rate=guess_rate, lapse_rate=lapse_rate, threshold=threshold, slope=slope)
    return params


@pytest.mark.parametrize('model', [ModelLogistic, ModelWeibull, ModelNormal])
def test_calculate_psi(model, designs, params):
    psi = EnginePsi(model=model(), designs=designs, params=params)

    len_design = int(np.prod([np.shape(des)[0] for des in designs.values()]))
    len_param = int(np.prod([np.shape(par)[0] for par in params.values()]))

    assert psi.p_obs.shape == (len_design, len_param)


@pytest.mark.parametrize('design_type', ['optimal', 'staircase', 'random'])
@pytest.mark.parametrize('model', [ModelLogistic, ModelWeibull, ModelNormal])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, designs, params, response):
    psi = EnginePsi(model=model(), designs=designs, params=params)
    d = psi.get_design(design_type)
    psi.update(d, response)


if __name__ == '__main__':
    pytest.main()
