r"""
**The choice under risk and ambiguity (CRA) task** [Levy2010]_ involves
preferential choice decisions in which the participant is asked to indicated
his/her preference between two options:

1. a reference option of either winning either a fixed amount of reward
   (:math:`R_F`) with a probability of 0.5 or winning none otherwise; and
2. a variable option of either winning a varying amount of reward
   (:math:`R_V`) with a varying probability (:math:`p_V`) and a varying
   ambiguity (:math:`A_V`) or winning none otherwise.

Further, the variable option comes in two types:

(a) *risky* type in which the winning probabilities are fully known to the
    participant; and
(b) *ambiguous* type in which the winning probabilities are only partially
    known to the participant.

The level of ambiguity (:math:`A_V`) in the latter type is varied between 0
(no ambiguity and thus fully known) and 1 (total ambiguity and thus fully
unknown).

.. [Levy2010] Levy, I., Snell, J., Nelson, A. J., Rustichini, A., & Glimcher,
   P. W. (2010). Neural Representation of Subjective Value Under
   Risk and Ambiguity. *Journal of Neurophysiology, 103* (2), 1036-1047.
"""
import numpy as np

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit, const_positive

__all__ = ['TaskCRA', 'ModelLinear', 'ModelExp', 'EngineCRA']


class TaskCRA(Task):
    def __init__(self):
        super(TaskCRA, self).__init__(
            name='CRA',
            key='cra',
            designs=['prob', 'ambig', 'r_var', 'r_fix'],
            responses=[0, 1]  # binary response
        )


class ModelLinear(Model):
    def __init__(self):
        super(ModelLinear, self).__init__(
            name='Linear',
            key='lin',
            task=TaskCRA(),
            params=['alpha', 'beta', 'gamma'],
            constraint={
                'gamma': const_positive,
            }
        )

    def compute(cls, prob, ambig, r_var, r_fix, alpha, beta, gamma):
        sv_var = np.power(r_var, alpha)
        sv_var = (prob - beta * np.divide(ambig, 2)) * sv_var
        sv_fix = .5 * np.power(r_fix, alpha)
        return inv_logit(gamma * (sv_var - sv_fix))


class ModelExp(Model):
    def __init__(self):
        super(ModelExp, self).__init__(
            name='Exponential',
            key='exp',
            task=TaskCRA(),
            params=['alpha', 'beta', 'gamma'],
            constraint={
                'gamma': const_positive,
            }
        )

    def compute(cls, prob, ambig, r_var, r_fix, alpha, beta, gamma):
        sv_var = np.power(r_var, alpha)
        sv_var = np.power(prob, 1 + beta * ambig) * sv_var
        sv_fix = .5 * np.power(r_fix, alpha)
        return inv_logit(gamma * (sv_var - sv_fix))


class EngineCRA(Engine):
    """ADO implementations for the choice under risk and ambiguity task"""

    def __init__(self, model, designs, params):
        if model not in [ModelLinear(), ModelExp()]:
            raise AssertionError(
                'Model should be adopy.tasks.cra.ModelLinear or '
                'adopy.tasks.cra.ModelExp.')

        super(EngineCRA, self).__init__(
            task=TaskCRA(),
            model=model,
            designs=designs,
            params=params
        )
