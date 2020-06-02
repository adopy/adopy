r"""
**Delay discounting** refers to the well-established finding that humans
tend to discount the value of a future reward such that the discount
progressively increases as a function of the receipt delay (Green & Myerson,
2004; Vincent, 2016).
In a typical **delay discounting (DD) task**, the participant is asked to
indicate his/her preference between two delayed options:
a smaller-sooner (SS) option (e.g., 8 dollars now) and
a larger-longer (LL) option (e.g., 50 dollars in 1 month).

References
----------
Green, L. and Myerson, J. (2004). A discounting framework for choice with
delayed and probabilistic rewards. *Psychological Bulletin, 130*, 769–792.

Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis
testing for delay discounting tasks. *Behavior Research Methods, 48*,
1608–1620.
"""
import numpy as np
from scipy.stats import bernoulli

from adopy.base import Engine, Task, Model
from adopy.functions import inv_logit

__all__ = [
    'TaskDD',
    'ModelExp',
    'ModelHyp',
    'ModelHPB',
    'ModelCOS',
    'ModelQH',
    'ModelDE',
    'EngineDD'
]


class TaskDD(Task):
    """
    The Task class for the delay discounting task.

    Design variables
        - ``t_ss`` (:math:`t_{SS}`) - delay of a SS option
        - ``t_ll`` (:math:`t_{LL}`) - delay of a LL option
        - ``r_ss`` (:math:`R_{SS}`) - amount of reward of a SS option
        - ``r_ll`` (:math:`R_{LL}`) - amount of reward of a LL option

    Responses
        - ``choice`` - 0 (choosing a SS option) or 1 (choosing a LL option)

    Examples
    --------
    >>> from adopy.tasks.ddt import TaskDD
    >>> task = TaskDD()
    >>> task.designs
    ['t_ss', 't_ll', 'r_ss', 'r_ll']
    >>> task.responses
    ['choice']
    """

    def __init__(self):
        super(TaskDD, self).__init__(
            name='Delay discounting task',
            designs=['t_ss', 't_ll', 'r_ss', 'r_ll'],
            responses=['choice']  # Binary response
        )


class ModelExp(Model):
    r"""
    The exponential model for the delay discounting task (Samuelson, 1937).

    .. math::

        \begin{align}
            D(t) &= e^{-rt} \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``r`` (:math:`r`) - discounting parameter (:math:`r > 0`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    Samuelson, P. A. (1937). A note on measurement of utility.
    *The review of economic studies, 4* (2), 155–161.

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelExp
    >>> model = ModelExp()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['r', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Exponential model for the DD task',
            task=TaskDD(),
            params=['r', 'tau'])
        super(ModelExp, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, r, tau):
        def discount(delay):
            return np.exp(-delay * r)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class ModelHyp(Model):
    r"""
    The hyperbolic model for the delay discounting task (Mazur, 1987).

    .. math::

        \begin{align}
            D(t) &= \frac{1}{1 + kt} \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``k`` (:math:`k`) - discounting parameter (:math:`k > 0`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement.
    *Commons, ML.;Mazur, JE.; Nevin, JA*, 55–73.

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelHyp
    >>> model = ModelHyp()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['k', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Hyperbolic model for the DD task',
            task=TaskDD(),
            params=['k', 'tau'])
        super(ModelHyp, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, k, tau):
        def discount(delay):
            return np.divide(1, 1 + k * delay)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class ModelHPB(Model):
    r"""
    The hyperboloid model for the delay discounting task (Green & Myerson, 2004).

    .. math::

        \begin{align}
            D(t) &= \frac{1}{(1 + kt)^s} \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``k`` (:math:`k`) - discounting parameter (:math:`k > 0`)
        - ``s`` (:math:`s`) - scale parameter (:math:`s > 0`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    Green, L. and Myerson, J. (2004). A discounting framework for choice with
    delayed and probabilistic rewards.
    *Psychological Bulletin, 130*, 769–792.

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelHPB
    >>> model = ModelHPB()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['k', 's', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Hyperboloid model for the DD task',
            task=TaskDD(),
            params=['k', 's', 'tau'])
        super(ModelHPB, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, k, s, tau):
        def discount(delay):
            return np.divide(1, np.power(1 + k * delay, s))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class ModelCOS(Model):
    r"""
    The constant sensitivity model for the delay discounting task
    (Ebert & Prelec, 2007).

    .. math::

        \begin{align}
            D(t) &= \exp[-(rt)^s] \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``r`` (:math:`r`) - discounting parameter (:math:`r > 0`)
        - ``s`` (:math:`s`) - scale parameter (:math:`s > 0`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    Ebert, J. E. and Prelec, D. (2007). The fragility of time: Time-insensitivity
    and valuation of thenear and far future. *Management science, 53* (9), 1423–1438.

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelCOS
    >>> model = ModelCOS()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['r', 's', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Constant Sensitivity model for the DD task',
            task=TaskDD(),
            params=['r', 's', 'tau'])
        super(ModelCOS, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, r, s, tau):
        def discount(delay):
            return np.exp(-np.power(delay * r, s))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class ModelQH(Model):
    r"""
    The quasi-hyperbolic model (or Beta-Delta model) for the delay discounting task
    (Laibson, 1997).

    .. math::

        \begin{align}
            D(t) &= \begin{cases}
                1 & \text{if } t = 0 \\
                \beta \delta ^ t & \text{if } t > 0
            \end{cases} \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``beta`` (:math:`\beta`) - constant rate (:math:`0 < \beta < 1`)
        - ``delta`` (:math:`\delta`) - constant rate (:math:`0 < \delta < 1`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    Laibson, D. (1997). Golden eggs and hyperbolic discounting.
    *The Quarterly Journal of Economics*, 443–477

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelQH
    >>> model = ModelQH()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['beta', 'delta', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Quasi-Hyperbolic model for the DD task',
            task=TaskDD(),
            params=['beta', 'delta', 'tau'])
        super(ModelQH, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, beta, delta, tau):
        def discount(delay):
            return np.where(delay == 0,
                            np.ones_like(beta * delta * delay),
                            beta * np.power(delta, delay))

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class ModelDE(Model):
    r"""
    The double exponential model for the delay discounting task (McClure et al., 2007).

    .. math::

        \begin{align}
            D(t) &= \omega e^{-rt} + (1 - \omega) e^{-st} \\
            V_{LL} &= R_{LL} \cdot D(t_{LL}) \\
            V_{SS} &= R_{SS} \cdot D(t_{SS}) \\
            P(LL\, over \, SS) &= \frac{1}{1 + \exp [-\tau (V_{LL} - V_{SS})]}
        \end{align}

    Model parameters
        - ``omega`` (:math:`r`) - weight parameter (:math:`0 < \omega < 1`)
        - ``r`` (:math:`r`) - discounting rate (:math:`r > 0`)
        - ``s`` (:math:`s`) - discounting rate (:math:`s > 0`)
        - ``tau`` (:math:`\tau`) - inverse temperature (:math:`\tau > 0`)

    References
    ----------
    McClure, S. M., Ericson, K. M., Laibson, D. I., Loewenstein, G., and Cohen, J. D.
    (2007). Time discounting for primary rewards. *Journal of neuroscience, 27* (21),
    5796–5804.

    Examples
    --------
    >>> from adopy.tasks.ddt import ModelDE
    >>> model = ModelDE()
    >>> model.task
    Task('DDT', designs=['t_ss', 't_ll', 'r_ss', 'r_ll'], responses=[0, 1])
    >>> model.params
    ['omega', 'r', 's', 'tau']
    """

    def __init__(self):
        args = dict(
            name='Double Exponential model for the DD task',
            task=TaskDD(),
            params=['omega', 'r', 's', 'tau'])
        super(ModelDE, self).__init__(**args)

    def compute(self, choice, t_ss, t_ll, r_ss, r_ll, omega, r, s, tau):
        def discount(delay):
            return omega * np.exp(-delay * r) + \
                (1 - omega) * np.exp(-delay * s)

        v_ss = r_ss * discount(t_ss)
        v_ll = r_ll * discount(t_ll)

        # Probability to choose an option with late and large rewards.
        p_obs = inv_logit(tau * (v_ll - v_ss))
        return bernoulli.logpmf(choice, p_obs)


class EngineDD(Engine):
    """
    The Engine class for the delay discounting task.
    It can be only used for :py:class:`TaskDD`.
    """

    def __init__(self, model, grid_design, grid_param, **kwargs):
        if not isinstance(model.task, TaskDD):
            raise RuntimeError(
                'The model should be implemented for the DD task.')

        grid_response = {'choice': [0, 1]}

        super(EngineDD, self).__init__(
            task=model.task,
            model=model,
            grid_design=grid_design,
            grid_param=grid_param,
            grid_response=grid_response,
            **kwargs
        )
