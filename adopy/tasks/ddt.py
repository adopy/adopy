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

1. Exponential
2. Hyperbolic
3. Generalized Hyperbolic
4. Quasi-hyperbolic
5. Double exponential
6. Constant sensitivity


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
    'TaskDDT', 'ModelExp', 'ModelHyperbolic', 'ModelGeneralizedHyperbolic',
    'ModelQuasiHyperbolic', 'ModelDoubleExp', 'ModelCS', 'EngineDDT'
]


class TaskDDT(Task):
    def __init__(self):
        args = dict(name='DDT',
                    key='ddt',
                    design=['d_soon', 'd_late', 'a_soon', 'a_late'])
        super(TaskDDT, self).__init__(**args)


class ModelExp(Model):
    def __init__(self):
        args = dict(
            name='Exponential',
            key='exp',
            task=TaskDDT(),
            param=['tau', 'r'],
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


class ModelHyperbolic(Model):
    def __init__(self):
        args = dict(
            name='Hyperbolic',
            key='hyp',
            task=TaskDDT(),
            param=['tau', 'k'],
            constraint={
                'tau': const_positive,
                'k': const_positive,
            })
        super(ModelHyperbolic, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, k):
        def discount(delay):
            return np.divide(1, 1 + k * delay)

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelGeneralizedHyperbolic(Model):
    def __init__(self):
        args = dict(
            name='Generalized Hyperbolic',
            key='ghyp',
            task=TaskDDT(),
            param=['tau', 'k', 's'],
            constraint={
                'tau': const_positive,
                'k': const_positive,
            })
        super(ModelGeneralizedHyperbolic, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, k, s):
        def discount(delay):
            return np.divide(1, np.power(1 + k * delay, s))

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelQuasiHyperbolic(Model):
    def __init__(self):
        args = dict(
            name='Quasi-Hyperbolic',
            key='qhyp',
            task=TaskDDT(),
            param=['tau', 'beta', 'delta'],
            constraint={
                'tau': const_positive,
                'beta': const_01,
                'delta': const_01,
            })
        super(ModelQuasiHyperbolic, self).__init__(**args)

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


class ModelDoubleExp(Model):
    def __init__(self):
        args = dict(
            name='Double Exponential',
            key='dexp',
            task=TaskDDT(),
            param=['tau', 'omega', 'r', 's'],
            constraint={
                'tau': const_positive,
                'omega': const_01,
                'r': const_positive,
                's': const_positive,
            })
        super(ModelDoubleExp, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, omega, r, s):
        def discount(delay):
            return omega * np.exp(-delay * r) + \
                (1 - omega) * np.exp(-delay * s)

        v_ss = a_soon * discount(d_soon)
        v_ll = a_late * discount(d_late)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return p_obs


class ModelCS(Model):
    def __init__(self):
        args = dict(
            name='Constant Sensitivity',
            key='cs',
            task=TaskDDT(),
            param=['tau', 'r', 's'],
            constraint={
                'tau': const_positive,
                'r': const_positive,
                's': const_positive,
            })
        super(ModelCS, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, r, s):
        def discount(delay):
            return np.exp(-np.power(delay * r, s))

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
            ModelHyperbolic(),
            ModelHyperbolic(),
            ModelQuasiHyperbolic(),
            ModelGeneralizedHyperbolic(),
            ModelDoubleExp(),
            ModelCS()
        ]

        args = {
            'task': TaskDDT(),
            'model': model,
            'designs': designs,
            'params': params,
            'y_obs': np.array([0., 1.]),  # Binary response
        }
        super(EngineDDT, self).__init__(**args)
