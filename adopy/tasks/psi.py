r"""
**Psychometric function** is to figure out whether a subject can perceive a
signal with varying levels of magnitude. The function has one design variable
for the *intensity* of a stimulus, :math:`x`; the model has four model
parameters: *guess rate* :math:`\gamma`, *lapse rate* :math:`\delta`,
*threshold* :math:`\alpha`, and *slope* :math:`\beta`.
"""
import numpy as np
from scipy.stats import bernoulli, norm, gumbel_l

from adopy.base import Engine, Task, Model
from adopy.functions import (inv_logit, get_random_design_index,
                             get_nearest_grid_index, const_positive, const_01)
from adopy.types import integer_like

__all__ = [
    'Task2AFC', 'ModelLogistic', 'ModelWeibull', 'ModelProbit', 'EnginePsi'
]


class Task2AFC(Task):
    """
    The Task class for a simple 2-Alternative Forced Choice (2AFC) task
    with a single design variable, the magnitude of a stimulus.

    Design variables
        - ``stimulus`` (:math:`x`) - magnitude of a stimulus

    Responses
        - ``choice`` (:math:`y`) - index of the chosen option

    Examples
    --------
    >>> from adopy.tasks.psi import Task2AFC
    >>> task = Task2AFC()
    >>> task.designs
    ['stimulus']
    >>> task.responses
    [0, 1]
    """

    def __init__(self):
        super(Task2AFC, self).__init__(
            name='2-alternative forced choice task',
            designs=['stimulus'],
            responses=['choice']
        )


class _ModelPsi(Model):
    def __init__(self, name):
        super(_ModelPsi, self).__init__(
            name=name,
            task=Task2AFC(),
            params=['threshold', 'slope', 'guess_rate', 'lapse_rate'],
        )

    def _compute_prob(self, func, st, th, sl, gr, lr):
        return gr + (1 - gr - lr) * func(sl * (st - th))


class ModelLogistic(_ModelPsi):
    r"""
    The psychometric function using logistic function.

    .. math::

        \begin{align}
            F(x; \alpha, \beta) &= \frac{1}{1 + \exp\left(-\beta (x - \mu) \right)} \\
            \Psi \left( x \mid \alpha, \beta, \gamma, \delta \right)
                &= \gamma + (1 - \gamma - \delta) \cdot F(x; \alpha, \beta)
        \end{align}

    Model parameters
        - ``threshold`` (:math:`\alpha`) - indifference point where the probability
          of a positive response equals to 0.5.
        - ``slope`` (:math:`\beta`) - slope of a tangent line on the threshold point
          (:math:`\beta > 0`)
        - ``guess_rate`` (:math:`\gamma`) - leftward asymptote of the psychometric
          function (:math:`0 < \gamma < 1`)
        - ``lapse_rate`` (:math:`\delta`) - rightward asymptote of the psychometric
          function(:math:`0 < \delta < 1`)

    Examples
    --------
    >>> from adopy.tasks.psi import ModelLogistic
    >>> model = ModelLogistic()
    >>> model.task
    Task('2AFC', designs=['stimulus'], responses=[0, 1])
    >>> model.params
    ['threshold', 'slope', 'guess_rate', 'lapse_rate']
    """

    def __init__(self):
        super(ModelLogistic, self).__init__(
            name='Logistic model for 2AFC tasks')

    def compute(self, choice, stimulus, guess_rate, lapse_rate, threshold, slope):
        p_obs = self._compute_prob(
            inv_logit, stimulus, threshold, slope, guess_rate, lapse_rate)
        return bernoulli.logpmf(choice, p_obs)


class ModelWeibull(_ModelPsi):
    r"""
    The psychometric function using log Weibull (Gumbel) cumulative distribution function.

    .. math::

        \begin{align}
            F(x; \mu, \beta) &= CDF_\text{Gumbel_l}
                \left( \beta (x - \mu) \right) \\
            \Psi \left( x \mid \alpha, \beta, \gamma, \delta \right)
                &= \gamma + (1 - \gamma - \delta) \cdot F(x; \alpha, \beta)
        \end{align}

    Model parameters
        - ``threshold`` (:math:`\alpha`) - indifference point where the probability
          of a positive response equals to 0.5.
        - ``slope`` (:math:`\beta`) - slope of a tangent line on the threshold point
          (:math:`\beta > 0`)
        - ``guess_rate`` (:math:`\gamma`) - leftward asymptote of the psychometric
          function (:math:`0 < \gamma < 1`)
        - ``lapse_rate`` (:math:`\delta`) - rightward asymptote of the psychometric
          function(:math:`0 < \delta < 1`)

    Examples
    --------
    >>> from adopy.tasks.psi import ModelLogistic
    >>> model = ModelLogistic()
    >>> model.task
    Task('2AFC', designs=['stimulus'], responses=[0, 1])
    >>> model.params
    ['threshold', 'slope', 'guess_rate', 'lapse_rate']
    """

    def __init__(self):
        super(ModelWeibull, self).__init__(
            name='Weibull model for 2AFC tasks')

    def compute(self, choice, stimulus, guess_rate, lapse_rate, threshold, slope):
        p_obs = self._compute_prob(
            gumbel_l.cdf, stimulus, threshold, slope, guess_rate, lapse_rate)
        return bernoulli.logpmf(choice, p_obs)


class ModelProbit(_ModelPsi):
    r"""
    The psychometric function using Probit function
    (Normal cumulative distribution function).

    .. math::

        \begin{align}
            F(x; \mu, \beta) &= CDF_\text{Normal}
                \left( \beta (x - \mu) \right) \\
            \Psi \left( x \mid \alpha, \beta, \gamma, \delta \right)
                &= \gamma + (1 - \gamma - \delta) \cdot F(x; \alpha, \beta)
        \end{align}

    Model parameters
        - ``threshold`` (:math:`\alpha`) - indifference point where the probability
          of a positive response equals to 0.5.
        - ``slope`` (:math:`\beta`) - slope of a tangent line on the threshold point
          (:math:`\beta > 0`)
        - ``guess_rate`` (:math:`\gamma`) - leftward asymptote of the psychometric
          function (:math:`0 < \gamma < 1`)
        - ``lapse_rate`` (:math:`\delta`) - rightward asymptote of the psychometric
          function(:math:`0 < \delta < 1`)

    Examples
    --------
    >>> from adopy.tasks.psi import ModelProbit
    >>> model = ModelProbit()
    >>> model.task
    Task('2AFC', designs=['stimulus'], responses=[0, 1])
    >>> model.params
    ['threshold', 'slope', 'guess_rate', 'lapse_rate']
    """

    def __init__(self):
        super(ModelProbit, self).__init__(
            name='Probit model for 2AFC tasks')

    def compute(self, choice, stimulus, guess_rate, lapse_rate, threshold, slope):
        p_obs = self._compute_prob(
            norm.cdf, stimulus, threshold, slope, guess_rate, lapse_rate)
        return bernoulli.logpmf(choice, p_obs)


class EnginePsi(Engine):
    """
    The Engine class for the psychometric function estimation.
    It can be only used for :py:class:`Task2AFC`.
    """

    def __init__(self, model, grid_design, grid_param, d_step: int = 1, **kwargs):
        if not isinstance(model.task, Task2AFC):
            raise RuntimeError(
                'The model should be implemented for the CRA task.')

        grid_response = {'choice': [0, 1]}

        super(EnginePsi, self).__init__(
            task=model.task,
            model=model,
            grid_design=grid_design,
            grid_param=grid_param,
            grid_response=grid_response,
            **kwargs
        )

        self.idx_opt = get_random_design_index(self.grid_design)
        self.y_obs_prev = 1
        self.d_step = d_step

    @property
    def d_step(self) -> int:
        r"""
        Step size on index to compute :math:`\Delta` for the staircase method.
        """
        return self._d_step

    @d_step.setter
    def d_step(self, value: integer_like):
        if not isinstance(value, (int, np.int)) or value <= 0:
            raise ValueError('d_step should be an positive integer.')
        self._d_step = int(value)

    def get_design(self, kind='optimal'):
        r"""Choose a design with a given type.

        - :code:`optimal`: an optimal design :math:`d^*` that maximizes
          the information for fitting model parameters.

        - :code:`staircase`: Choose the stimulus :math:`s` as below:

            .. math::
                x_t = \begin{cases}
                    x_{t-1} - \Delta & \text{if } y_{t-1} = 1 \\
                    x_{t-1} + 2 \Delta & \text{if } y_{t-1} = 0
                \end{cases}

          where :math:`\Delta` is determined by ``d_step`` which is the
          step-size change on the grid index.

        - :code:`random`: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'staircase', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design
            A chosen design vector
        """
        assert kind in {'optimal', 'staircase', 'random'}

        self._update_mutual_info()

        if kind == 'optimal':
            ret = self.grid_design.iloc[np.argmax(self.mutual_info)]

        elif kind == 'staircase':
            if self.y_obs_prev == 1:
                idx = max(0, self.idx_opt - self.d_step)
            else:
                idx = min(len(self.grid_design) - 1,
                          self.idx_opt + (self.d_step * 2))

            ret = self.grid_design.iloc[np.int(idx)]

        elif kind == 'random':
            ret = self.grid_design.iloc[get_random_design_index(
                self.grid_design)]

        else:
            raise RuntimeError('An invalid kind of design: "{}".'.format(type))

        self.idx_opt = get_nearest_grid_index(ret, self.grid_design)

        return ret

    def update(self, design, response):
        super(EnginePsi, self).update(design, response)

        # Store the previous response for staircase
        self.y_obs_prev = response
