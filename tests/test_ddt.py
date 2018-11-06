from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from adopy.tasks.ddt import (ModelExp, ModelHyperbolic, ModelGeneralizedHyperbolic, ModelQuasiHyperbolic,
                             ModelDoubleExp, ModelCS, EngineDDT)

N_GRID = 7


def make_grid(start, end, n):
    return np.linspace(start, end, n + 1, endpoint=False)[1:]


@pytest.fixture()
def designs():
    # Amounts of rewards
    am_soon = [8, 12, 15, 17, 19, 22]
    am_late = [12, 15, 17, 19, 22, 23]

    amounts = np.vstack([(ams, aml) for ams in am_soon for aml in am_late if ams < aml])

    # Delays
    d_soon = [0, 1, 2, 3, 5, 10, 20, 40]
    d_late = [1, 2, 3, 5, 10, 20, 40, 80]

    delays = np.vstack([(ds, dl) for ds in d_soon for dl in d_late if ds < dl])

    designs = {('d_soon', 'd_late'): delays, ('a_soon', 'a_late'): amounts}
    return designs


@pytest.mark.parametrize('model, params', [
    (ModelExp, dict(tau=make_grid(0, 5, N_GRID), r=make_grid(0, 2, N_GRID))),
    (ModelHyperbolic, dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID))),
    (ModelGeneralizedHyperbolic,
     dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID), s=make_grid(-1, 1, N_GRID))),
    (ModelQuasiHyperbolic,
     dict(tau=make_grid(0, 5, N_GRID), beta=make_grid(0, 1, N_GRID), delta=make_grid(0, 1, N_GRID))),
    (ModelDoubleExp,
     dict(
         tau=make_grid(0, 5, N_GRID),
         omega=make_grid(0, 1, N_GRID),
         r=make_grid(0, 2, N_GRID),
         s=make_grid(0, 2, N_GRID))),
    (ModelCS, dict(tau=np.linspace(0, 5, N_GRID), r=np.linspace(0, 2, N_GRID), s=np.linspace(0, 2, N_GRID))),
])
def test_calculate_psi(model, designs, params):
    ddt = EngineDDT(model=model(), designs=designs, params=params)

    len_design = int(np.prod([np.shape(des)[0] for des in designs.values()]))
    len_param = int(np.prod([np.shape(par)[0] for par in params.values()]))

    assert ddt.p_obs.shape == (len_design, len_param)


@pytest.mark.parametrize('design_type', ['optimal', 'random'])
@pytest.mark.parametrize(
    'model, params',
    [
        (ModelExp, dict(tau=make_grid(0, 5, N_GRID), r=make_grid(0, 2, N_GRID))),
        (ModelHyperbolic, dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID))),
        (ModelGeneralizedHyperbolic,
         dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID), s=make_grid(-1, 1, N_GRID))),
        (ModelQuasiHyperbolic,
         dict(tau=make_grid(0, 5, N_GRID), beta=make_grid(0, 1, N_GRID), delta=make_grid(0, 1, N_GRID))),
        (ModelDoubleExp,
         dict(
             tau=make_grid(0, 5, N_GRID),
             omega=make_grid(0, 1, N_GRID),
             r=make_grid(0, 2, N_GRID),
             s=make_grid(0, 2, N_GRID))),
        (ModelCS, dict(tau=np.linspace(0, 5, N_GRID), r=np.linspace(0, 2, N_GRID), s=np.linspace(0, 2, N_GRID))),
    ])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, designs, params, response):
    ddt = EngineDDT(model=model(), designs=designs, params=params)
    d = ddt.get_design(design_type)
    ddt.update(d, response)


if __name__ == '__main__':
    pytest.main()
