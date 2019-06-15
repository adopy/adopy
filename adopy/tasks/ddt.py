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

1. Exponential (`ModelExp`)
2. Hyperbolic (`ModelHyp`)
3. Hyperboloid (`ModelHPB`)
4. Quasi-hyperbolic (`ModelQH`)
5. Double exponential (`ModelDE`)
6. Constant sensitivity (`ModelCOS`)

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
            designs=['t_ss', 't_ll', 'r_ss', 'r_ll'],
            responses=[0, 1]  # Binary response
        )


class ModelExp(Model):
    def __init__(self):
        args = dict(
            name='Exponential',
            task=TaskDDT(),
            params=['r', 'tau'],
            constraint={
                'r': const_positive,
                'tau': const_positive,
            })
        super(ModelExp, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, r, tau):
        def discount(delay):
            return np.exp(-delay * r)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelHyp(Model):
    def __init__(self):
        args = dict(
            name='Hyperbolic',
            task=TaskDDT(),
            params=['k', 'tau'],
            constraint={
                'k': const_positive,
                'tau': const_positive,
            })
        super(ModelHyp, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, k, tau):
        def discount(delay):
            return np.divide(1, 1 + k * delay)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelHPB(Model):
    def __init__(self):
        args = dict(
            name='Hyperboloid',
            task=TaskDDT(),
            params=['k', 's', 'tau'],
            constraint={
                'k': const_positive,
                'tau': const_positive,
            })
        super(ModelHPB, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, k, s, tau):
        def discount(delay):
            return np.divide(1, np.power(1 + k * delay, s))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelCOS(Model):
    def __init__(self):
        args = dict(
            name='Constant Sensitivity',
            task=TaskDDT(),
            params=['r', 's', 'tau'],
            constraint={
                'r': const_positive,
                's': const_positive,
                'tau': const_positive,
            })
        super(ModelCOS, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, r, s, tau):
        def discount(delay):
            return np.exp(-np.power(delay * r, s))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelQH(Model):
    def __init__(self):
        args = dict(
            name='Quasi-Hyperbolic',
            task=TaskDDT(),
            params=['beta', 'delta', 'tau'],
            constraint={
                'beta': const_01,
                'delta': const_01,
                'tau': const_positive,
            })
        super(ModelQH, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, beta, delta, tau):
        def discount(delay):
            return np.where(delay == 0,
                            np.ones_like(beta * delta * delay),
                            beta * np.power(delta, delay))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelDE(Model):
    def __init__(self):
        args = dict(
            name='Double Exponential',
            task=TaskDDT(),
            params=['omega', 'r', 's', 'tau'],
            constraint={
                'omega': const_01,
                'r': const_positive,
                's': const_positive,
                'tau': const_positive,
            })
        super(ModelDE, self).__init__(**args)

    def compute(self, t_ss, t_ll, r_ss, r_ll, omega, r, s, tau):
        def discount(delay):
            return omega * np.exp(-delay * r) + \
                (1 - omega) * np.exp(-delay * s)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class EngineDDT(Engine):
    """ADO engine for delayed discounting task"""

    def __init__(self, model, designs, params):
        assert type(model) in [
            type(ModelExp()),
            type(ModelHyp()),
            type(ModelQH()),
            type(ModelHPB()),
            type(ModelDE()),
            type(ModelCOS()),
        ]

        super(EngineDDT, self).__init__(
            task=TaskDDT(),
            model=model,
            designs=designs,
            params=params
        )
