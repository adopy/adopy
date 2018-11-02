"""
===============================
Choice under Risk and Ambiguity
===============================

awefaw

References
----------

.. [Levy2010] Levy, I., Snell, J., Nelson, A. J., Rustichini, A., & Glimcher, P. W. (2010).
  Neural Representation of Subjective Value Under Risk and Ambiguity. *Journal of Neurophysiology, 103* (2), 1036–1047.

"""
from __future__ import absolute_import, division, print_function

import numpy as np

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit

__all__ = ['TaskCRA', 'ModelLinear', 'ModelExp', 'EngineCRA']


class TaskCRA(Task):
    def __init__(self):
        args = dict(name='CRA', key='cra', design=['prob', 'ambig', 'r_var', 'r_fix'])
        super(TaskCRA, self).__init__(**args)


class ModelLinear(Model):
    def __init__(self):
        args = dict(
            name='Linear',
            key='lin',
            task=TaskCRA(),
            param=['alpha', 'beta', 'gamma'],
            constraint={
                'gamma': lambda x: x >= 0,
            })
        super(ModelLinear, self).__init__(**args)

    def compute(cls, prob, ambig, r_var, r_fix, alpha, beta, gamma):
        # Calculate the subjective value of a variable option (risky or ambiguous).
        sv_var = np.power(r_var, alpha)
        sv_var = (prob - beta * np.divide(ambig, 2)) * sv_var

        # Calculate the subjective value of a reference option with a fixed probability of 0.5.
        sv_fix = .5 * np.power(r_fix, alpha)

        # Using a logistic function, compute the probability to choose the variable option.
        p_obs = inv_logit(gamma * (sv_var - sv_fix))
        return p_obs


class ModelExp(Model):
    def __init__(self):
        args = dict(
            name='Exponential',
            key='exp',
            task=TaskCRA(),
            param=['alpha', 'beta', 'gamma'],
            constraint={
                'gamma': lambda x: x >= 0,
            })
        super(ModelExp, self).__init__(**args)

    def compute(cls, prob, ambig, r_var, r_fix, alpha, beta, gamma):
        # Calculate the subjective value of a variable option (risky or ambiguous).
        sv_var = np.power(r_var, alpha)
        sv_var = np.power(prob, 1 + beta * ambig) * sv_var

        # Calculate the subjective value of a reference option with a fixed probability of 0.5.
        sv_fix = .5 * np.power(r_fix, alpha)

        # Using a logistic function, compute the probability to choose the variable option.
        p_obs = inv_logit(gamma * (sv_var - sv_fix))
        return p_obs


class EngineCRA(Engine):
    """ADO implementations for the choice under risk and ambiguity task"""

    def __init__(self, model, designs, params):
        assert model in [ModelLinear(), ModelExp()]

        args = {
            'task': TaskCRA(),
            'model': model,
            'designs': designs,
            'params': params,
            'y_obs': np.array([0., 1.]),  # Binary response
        }
        super(EngineCRA, self).__init__(**args)
