import numpy as np
import pytest

from adopy.tasks.cra import ModelLinear, ModelExp, EngineCRA


@pytest.fixture()
def grid_design():
    # p_var & a_var for risky & ambiguous trials
    pval = [.05, .10, .15, .20, .25, .30, .35, .40, .45]
    aval = [.125, .25, .375, .5, .625, .75]

    # risky trials: a_var fixed to 0
    pa_risky = [[p, 0] for p in pval]
    # ambiguous trials: p_var fixed to 0.5
    pa_ambig = [[0.5, a] for a in aval]
    pr_am = np.array(pa_risky + pa_ambig)

    # r_var & r_fix while r_var > r_fix
    rval = [10, 15, 21, 31, 45, 66, 97, 141, 206, 300]
    rewards = []
    for r_var in rval:
        for r_fix in rval:
            if r_var > r_fix:
                rewards.append([r_var, r_fix])
    rewards = np.array(rewards)

    return {('p_var', 'a_var'): pr_am, ('r_var', 'r_fix'): rewards}


@pytest.fixture()
def grid_param():
    alp = np.linspace(0, 3, 11)
    bet = np.linspace(-3, 3, 11)
    gam = np.linspace(0, 5, 11)
    params = dict(alpha=alp, beta=bet, gamma=gam)
    return params


@pytest.mark.parametrize('model', [ModelLinear, ModelExp])
def test_calculate_psi(model, grid_design, grid_param):
    cra = EngineCRA(model=model(),
                    grid_design=grid_design,
                    grid_param=grid_param)

    len_design = int(np.prod([np.shape(des)[0]
                              for des in grid_design.values()]))
    len_param = int(np.prod([np.shape(par)[0] for par in grid_param.values()]))

    assert cra.p_obs.shape == (len_design, len_param)


@pytest.mark.parametrize('design_type', ['optimal', 'random'])
@pytest.mark.parametrize('model', [ModelLinear, ModelExp])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, grid_design, grid_param, response):
    cra = EngineCRA(model=model(),
                    grid_design=grid_design,
                    grid_param=grid_param)
    d = cra.get_design(design_type)
    cra.update(d, response)


if __name__ == '__main__':
    pytest.main()
