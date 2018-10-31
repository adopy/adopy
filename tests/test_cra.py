from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy.functions import make_grid_matrix
from adopy.tasks.cra import ModelLinear, ModelExp, EngineCRA


@pytest.fixture()
def designs():
    # Define grids for the probability for rewarding and the ambiguity level
    ## For risky conditions
    pr_risky = np.linspace(0.0, 0.5, 5)
    am_risky = np.array(0).reshape(-1)

    ## For ambiguous conditions
    pr_ambig = np.array(0.5).reshape(-1)
    am_ambig = np.linspace(0.0, 0.75, 5)

    ## Make cartesian products for each condition
    pr_am_risky = np.squeeze(np.stack(np.meshgrid(pr_risky, am_risky), -1))
    pr_am_ambig = np.squeeze(np.stack(np.meshgrid(pr_ambig, am_ambig), -1))

    ## Merge two grids into one object
    pr_am = np.vstack([pr_am_risky[:-1, :], pr_am_ambig])

    # Define grids for rewards on each option
    r_var = np.round(np.logspace(np.log10(10), np.log10(250), 5, base=10))
    r_fix = np.round(np.logspace(np.log10(10), np.log10(125), 5, base=10))

    rs = np.vstack([(rv, rf) for rv in r_var for rf in r_fix if rv > rf])

    designs = make_grid_matrix({
        ('prob', 'ambig'): pr_am,
        ('r_var', 'r_fix'): rs
    })
    return {k: v.values for k, v in designs.iteritems()}


@pytest.fixture()
def params():
    alp = np.linspace(0.0, 2.0, 5)
    bet = np.linspace(-1.0, 2.0, 5)
    gam = np.linspace(0.0, 5.0, 5)
    params = dict(alpha=alp, beta=bet, gamma=gam)
    return params


@pytest.mark.parametrize('model', [ModelLinear, ModelExp])
def test_calculate_psi(model, designs, params):
    cra = EngineCRA(model=model(), designs=designs, params=params)

    len_design = int(np.prod([np.size(des) for des in designs.values()]))
    len_param = int(np.prod([np.size(par) for par in params.values()]))

    assert cra.p_obs.shape == (len_design, len_param)


@pytest.mark.parametrize('design_type', ['optimal', 'random'])
@pytest.mark.parametrize('model', [ModelLinear, ModelExp])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, designs, params, response):
    cra = EngineCRA(model=model(), designs=designs, params=params)
    d = cra.get_design(design_type)
    cra.update(d, response)


if __name__ == '__main__':
    pytest.main()
