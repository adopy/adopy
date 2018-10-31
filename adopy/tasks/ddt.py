from __future__ import absolute_import, division, print_function

import numpy as np

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit

__all__ = [
    'TaskDDT', 'ModelExp', 'ModelHyperbolic', 'ModelGeneralizedHyperbolic', 'ModelQuasiHyperbolic', 'ModelDoubleExp',
    'ModelCS', 'EngineDDT'
]


class TaskDDT(Task):
    def __init__(self):
        args = dict(name='DDT', key='ddt', design=['d_soon', 'd_late', 'a_soon', 'a_late'])
        super(TaskDDT, self).__init__(**args)


class ModelExp(Model):
    def __init__(self):
        args = dict(
            name='Exponential',
            key='exp',
            task=TaskDDT(),
            param=['tau', 'r'],
            constraint={
                'tau': lambda x: x > 0,
                'r': lambda x: x > 0,
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
                'tau': lambda x: x > 0,
                'k': lambda x: x > 0,
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
                'tau': lambda x: x > 0,
                'k': lambda x: x > 0,
            })
        super(ModelGeneralizedHyperbolic, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, k, s):
        def discount(delay):
            return np.exp(np.divide(1, 1 + k * delay), s)

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
                'tau': lambda x: x > 0,
                'beta': lambda x: 0 < x < 1,
            })
        super(ModelQuasiHyperbolic, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, beta, delta):
        def discount(delay):
            if delay == 0:
                return np.ones_like(beta * delta * delay)
            return beta * np.power(delta, delay)

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
                'tau': lambda x: x > 0,
                'omega': lambda x: 0 < x < 1,
                'r': lambda x: x > 0,
                's': lambda x: x > 0,
            })
        super(ModelDoubleExp, self).__init__(**args)

    def compute(cls, d_soon, d_late, a_soon, a_late, tau, omega, r, s):
        def discount(delay):
            return omega * np.exp(-delay * r) + (1 - omega) * np.exp(-delay * s)

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
                'tau': lambda x: x > 0,
                'r': lambda x: x > 0,
                's': lambda x: x > 0,
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
