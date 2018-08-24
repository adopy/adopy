from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy.psi import Psi
from adopy.functions import make_vector_shape


@pytest.fixture()
def stimuli():
    return np.linspace(20 * np.log10(.05), 20 * np.log10(400), 120)


@pytest.fixture()
def guess_rate():
    return 0.5


@pytest.fixture()
def lapse_rate():
    return 0.05


@pytest.fixture()
def threshold():
    return np.linspace(20 * np.log10(.1), 20 * np.log10(200), 100)


@pytest.fixture()
def slope():
    return np.linspace(0, 10, 11)[1:]


@pytest.mark.parametrize('func_type', ['l', 'w', 'g', 'n'])
def test_calculate_psi(func_type, stimuli, guess_rate, lapse_rate, threshold, slope):
    psi = Psi.compute_p_obs(
        func_type=func_type,
        stimulus=stimuli.reshape(make_vector_shape(3, 0)),
        guess_rate=guess_rate,
        lapse_rate=lapse_rate,
        threshold=threshold.reshape(make_vector_shape(3, 1)),
        slope=slope.reshape(make_vector_shape(3, 2)))

    assert psi.shape == (len(stimuli), len(threshold), len(slope))


@pytest.mark.parametrize('design_type', ['optimal', 'staircase', 'random'])
@pytest.mark.parametrize('func_type', ['l', 'w', 'n'])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, func_type, stimuli, guess_rate, lapse_rate, threshold, slope, response):
    o = Psi(func_type, stimuli, guess_rate, lapse_rate, threshold, slope)

    d = o.get_design(design_type)
    assert d in stimuli

    o.update(d, response)


if __name__ == '__main__':
    pytest.main()
