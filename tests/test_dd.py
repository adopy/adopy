import numpy as np
import pytest

from adopy.tasks.dd import (ModelExp, ModelHyp, ModelHPB, ModelQH,
                            ModelDE, ModelCOS, EngineDD)

N_GRID = 7


def make_grid(start, end, n):
    return np.linspace(start, end, n + 1, endpoint=False)[1:]


@pytest.fixture()
def grid_design():
    # Amounts of rewards
    am_soon = [8, 12, 15, 17, 19, 22]
    am_late = [12, 15, 17, 19, 22, 23]

    amounts = np.vstack([(ams, aml)
                         for ams in am_soon for aml in am_late if ams < aml])

    # Delays
    t_ss = [0, 1, 2, 3, 5, 10, 20, 40]
    t_ll = [1, 2, 3, 5, 10, 20, 40, 80]

    delays = np.vstack([(ds, dl) for ds in t_ss for dl in t_ll if ds < dl])

    designs = {('t_ss', 't_ll'): delays, ('r_ss', 'r_ll'): amounts}
    return designs


@pytest.mark.parametrize('design_type', ['optimal', 'random'])
@pytest.mark.parametrize('model, grid_param', [
    (ModelExp, dict(tau=make_grid(0, 5, N_GRID), r=make_grid(0, 2, N_GRID))),
    (ModelHyp, dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID))),
    (ModelHPB, dict(tau=make_grid(0, 5, N_GRID), k=make_grid(0, 2, N_GRID),
                    s=make_grid(-1, 1, N_GRID))),
    (ModelQH, dict(tau=make_grid(0, 5, N_GRID), beta=make_grid(0, 1, N_GRID),
                   delta=make_grid(0, 1, N_GRID))),
    (ModelDE, dict(tau=make_grid(0, 5, N_GRID), omega=make_grid(0, 1, N_GRID),
                   r=make_grid(0, 2, N_GRID), s=make_grid(0, 2, N_GRID))),
    (ModelCOS, dict(tau=np.linspace(0, 5, N_GRID), r=np.linspace(0, 2, N_GRID),
                    s=np.linspace(0, 2, N_GRID))),
])
@pytest.mark.parametrize('response', [0, 1])
def test_classes(design_type, model, grid_design, grid_param, response):
    ddt = EngineDD(model=model(),
                   grid_design=grid_design, grid_param=grid_param)
    d = ddt.get_design(design_type)
    ddt.update(d, response)


if __name__ == '__main__':
    pytest.main()
