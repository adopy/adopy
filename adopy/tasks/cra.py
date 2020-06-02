r"""
**The choice under risk and ambiguity task** (CRA; Levy et al., 2010) involves
preferential choice decisions in which the participant is asked to indicated
his/her preference between two options:

1. **A fixed (or reference) option** of either winning a fixed amount of reward
   (:math:`R_F`, ``r_fix``) with a fixed probability of 0.5 or
   winning none otherwise; and
2. **A variable option** of either winning a varying amount of reward
   (:math:`R_V`, ``r_var``) with a varying probability (:math:`p_V`, ``p_var``)
   and a varying level of ambiguity (:math:`A_V`, ``a_var``) or
   winning none otherwise.

Further, the variable option comes in two types:

(a) *risky* type in which the winning probabilities are fully known to the
    participant; and
(b) *ambiguous* type in which the winning probabilities are only partially
    known to the participant.

The level of ambiguity (:math:`A_V`) is varied between 0 (no ambiguity and thus
fully known) and 1 (total ambiguity and thus fully unknown).

References
----------
Levy, I., Snell, J., Nelson, A. J., Rustichini, A., & Glimcher, P. W. (2010).
Neural Representation of Subjective Value Under Risk and Ambiguity.
*Journal of Neurophysiology, 103* (2), 1036-1047.
"""
import numpy as np
from scipy.stats import bernoulli

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit

__all__ = ['TaskCRA', 'ModelLinear', 'ModelExp', 'EngineCRA']


class TaskCRA(Task):
    """
    The Task class for the choice under risk and ambiguity task (Levy et al.,
    2010).

    Design variables
        - ``p_var`` (:math:`p_V`) - probability to win of a variable option
        - ``a_var`` (:math:`A_V`) - level of ambiguity of a variable option
        - ``r_var`` (:math:`R_V`) - amount of reward of a variable option
        - ``r_fix`` (:math:`R_F`) - amount of reward of a fixed option

    Responses
        - ``choice`` - 0 (choosing a fixed option) or
          1 (choosing a variable option)

    Examples
    --------
    >>> from adopy.tasks.cra import TaskCRA
    >>> task = TaskCRA()
    >>> task.designs
    ['p_var', 'a_var', 'r_var', 'r_fix']
    >>> task.responses
    ['choice']
    """

    def __init__(self):
        super(TaskCRA, self).__init__(
            name='Choice under risk and ambiguity task',
            designs=['p_var', 'a_var', 'r_var', 'r_fix'],
            responses=['choice']  # binary response
        )


class ModelLinear(Model):
    r"""
    The linear model for the CRA task (Levy et al., 2010).

    .. math::

        \begin{align}
            U_F &= 0.5 \cdot (R_F)^\alpha \\
            U_V &= \left[ p_V - \beta \cdot \frac{A_V}{2} \right] \cdot (R_V)^\alpha \\
            P(V\, over \, F) &= \frac{1}{1 + \exp [-\gamma (U_V - U_F)]}
        \end{align}

    Model parameters
        - ``alpha`` (:math:`\alpha`) - risk attitude parameter (:math:`\alpha > 0`)
        - ``beta`` (:math:`\beta`) - ambiguity attitude parameter
        - ``gamma`` (:math:`\gamma`) - inverse temperature (:math:`\gamma > 0`)

    References
    ----------
    Levy, I., Snell, J., Nelson, A. J., Rustichini, A., & Glimcher,
    P. W. (2010). Neural Representation of Subjective Value Under
    Risk and Ambiguity. *Journal of Neurophysiology, 103* (2), 1036-1047.

    Examples
    --------
    >>> from adopy.tasks.cra import ModelExp
    >>> model = ModelExp()
    >>> model.task
    Task('CRA', designs=['p_var', 'a_var', 'r_var', 'r_fix'], responses=[0, 1])
    >>> model.params
    ['alpha', 'beta', 'gamma']
    """

    def __init__(self):
        super(ModelLinear, self).__init__(
            name='Linear model for the CRA task',
            task=TaskCRA(),
            params=['alpha', 'beta', 'gamma']
        )

    def compute(self, choice, p_var, a_var, r_var, r_fix, alpha, beta, gamma):
        sv_var = np.power(r_var, alpha)
        sv_var = (p_var - beta * np.divide(a_var, 2)) * sv_var
        sv_fix = .5 * np.power(r_fix, alpha)
        p_obs = inv_logit(gamma * (sv_var - sv_fix))
        return bernoulli.logpmf(choice, p_obs)


class ModelExp(Model):
    r"""
    The exponential model for the CRA task (Hsu et al., 2005).

    .. math::

        \begin{align}
            U_F &= 0.5 \cdot (R_F)^\alpha \\
            U_V &= (p_V) ^ {(1 + \beta \cdot A_V)} \cdot (R_V)^\alpha \\
            P(V\, over \, F) &= \frac{1}{1 + \exp [-\gamma (U_V - U_F)]}
        \end{align}

    Model parameters
        - ``alpha`` (:math:`\alpha`) - risk attitude parameter (:math:`\alpha > 0`)
        - ``beta`` (:math:`\beta`) - ambiguity attitude parameter
        - ``gamma`` (:math:`\gamma`) - inverse temperature (:math:`\gamma > 0`)

    References
    ----------
    Hsu, Y.-F., Falmagne, J.-C., and Regenwetter, M. (2005).
    The tuning in-and-out model: a randomwalk and its application to
    presidential election surveys.
    *Journal of Mathematical Psychology, 49*, 276–289.

    Examples
    --------
    >>> from adopy.tasks.cra import ModelLinear
    >>> model = ModelLinear()
    >>> model.task
    Task('CRA', designs=['p_var', 'a_var', 'r_var', 'r_fix'], responses=[0, 1])
    >>> model.params
    ['alpha', 'beta', 'gamma']
    """

    def __init__(self):
        super(ModelExp, self).__init__(
            name='Exponential model for the CRA task',
            task=TaskCRA(),
            params=['alpha', 'beta', 'gamma']
        )

    def compute(self, choice, p_var, a_var, r_var, r_fix, alpha, beta, gamma):
        sv_var = np.power(r_var, alpha)
        sv_var = np.power(p_var, 1 + beta * a_var) * sv_var
        sv_fix = .5 * np.power(r_fix, alpha)
        p_obs = inv_logit(gamma * (sv_var - sv_fix))
        return bernoulli.logpmf(choice, p_obs)


class EngineCRA(Engine):
    """
    The Engine class for the CRA task. It can be only used for :py:class:`TaskCRA`.
    """

    def __init__(self, model, grid_design, grid_param, **kwargs):
        if not isinstance(model.task, TaskCRA):
            raise RuntimeError(
                'The model should be implemented for the CRA task.')

        grid_response = {'choice': [0, 1]}

        super(EngineCRA, self).__init__(
            task=model.task,
            model=model,
            grid_design=grid_design,
            grid_param=grid_param,
            grid_response=grid_response,
            **kwargs,
        )
