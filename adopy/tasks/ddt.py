r"""
**Delay discounting** refers to the well-established finding that humans
tend to discount the value of a future reward such that the discount
progressively increases as a function of the receipt delay
[Green2004]_ [Vincent2016]_.
In a typical **delay discounting (DD) task**, the participant is asked to
indicate his/her preference between two delayed options:
a smaller-sooner (SS) option (e.g., 10 dollars tomorrow) and
a larger-longer (LL) option (e.g., 50 dollars in two weeks)

We provides six models that had been compared in a previous paper
[Cavagnaro2016]_:

1. Exponential (`adopy.tasks.ddt.ModelExp`)
2. Hyperbolic (`adopy.tasks.ddt.ModelHyp`)
3. Hyperboloid (`adopy.tasks.ddt.ModelHPB`)
4. Quasi-hyperbolic (`adopy.tasks.ddt.ModelQH`)
5. Double exponential (`adopy.tasks.ddt.ModelDE`)
6. Constant sensitivity (`adopy.tasks.ddt.ModelCOS`)

.. [Green2004]
   Green, L. and Myerson, J. (2004). A discounting framework for choice with
   delayed and probabilistic rewards. *Psychological Bulletin, 130*, 769–792.

.. [Vincent2016]
   Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis
   testing for delay discounting tasks. *Behavior Research Methods, 48*,
   1608–1620.

.. [Cavagnaro2016]
    Cavagnaro, D. R., Aranovich, G. J., McClure, S. M., Pitt, M. A., &
    Myung, J. I. (2016). On the functional form of temporal discounting:
    An optimized adaptive test. *Journal of risk and uncertainty, 52* (3),
    233-254.
"""
import numpy as np

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit, const_positive, const_01

__all__ = [
    'TaskDDT',
    'ModelExp',
    'ModelHyp',
    'ModelHPB',
    'ModelCOS',
    'ModelQH',
    'ModelDE',
    'EngineDDT'
]


class TaskDDT(Task):
    def __init__(self):
        super(TaskDDT, self).__init__(
            name='DDT',
            key='ddt',
            designs=['d_soon', 'd_late', 'a_soon', 'a_late'],
            responses=[0, 1]  # Binary response
        )


class ModelExp(Model):
    def __init__(self):
        args = dict(
            name='Exponential',
            key='exp',
            task=TaskDDT(),
            params=['tau', 'r'],
            constraint={
                'tau': const_positive,
                'r': const_positive,
            })
        super(ModelExp, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, r):
        def discount(delay):
            return np.exp(-delay * r)

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelHyp(Model):
    def __init__(self):
        args = dict(
            name='Hyperbolic',
            key='hyp',
            task=TaskDDT(),
            params=['tau', 'k'],
            constraint={
                'tau': const_positive,
                'k': const_positive,
            })
        super(ModelHyp, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, k):
        def discount(delay):
            return np.divide(1, 1 + k * delay)

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelHPB(Model):
    def __init__(self):
        args = dict(
            name='Hyperboloid',
            key='hpb',
            task=TaskDDT(),
            params=['tau', 'k', 's'],
            constraint={
                'tau': const_positive,
                'k': const_positive,
            })
        super(ModelHPB, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, k, s):
        def discount(delay):
            return np.divide(1, np.power(1 + k * delay, s))

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelCOS(Model):
    def __init__(self):
        args = dict(
            name='Constant Sensitivity',
            key='cs',
            task=TaskDDT(),
            params=['tau', 'r', 's'],
            constraint={
                'tau': const_positive,
                'r': const_positive,
                's': const_positive,
            })
        super(ModelCOS, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, r, s):
        def discount(delay):
            return np.exp(-np.power(delay * r, s))

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelQH(Model):
    def __init__(self):
        args = dict(
            name='Quasi-Hyperbolic',
            key='qhyp',
            task=TaskDDT(),
            params=['tau', 'beta', 'delta'],
            constraint={
                'tau': const_positive,
                'beta': const_01,
                'delta': const_01,
            })
        super(ModelQH, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, beta, delta):
        def discount(delay):
            return np.where(delay == 0,
                            np.ones_like(beta * delta * delay),
                            beta * np.power(delta, delay))

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelDE(Model):
    def __init__(self):
        args = dict(
            name='Double Exponential',
            key='dexp',
            task=TaskDDT(),
            params=['tau', 'omega', 'r', 's'],
            constraint={
                'tau': const_positive,
                'omega': const_01,
                'r': const_positive,
                's': const_positive,
            })
        super(ModelDE, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, omega, r, s):
        def discount(delay):
            return omega * np.exp(-delay * r) + \
                (1 - omega) * np.exp(-delay * s)

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class EngineDDT(Engine):
    """ADO engine for delayed discounting task"""

    def __init__(self, model, designs, params):
        assert model in [
            ModelExp(),
            ModelHyp(),
            ModelQH(),
            ModelHPB(),
            ModelDE(),
            ModelCOS()
        ]

        super(EngineDDT, self).__init__(
            task=TaskDDT(),
            model=model,
            designs=designs,
            params=params
        )
