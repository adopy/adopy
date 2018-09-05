from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.stats import norm, gumbel_l
from scipy.special import logsumexp

from .generics import ADOGeneric
from .functions import inv_logit, log_lik_bern
from .functions import get_random_design_index, get_nearest_grid_index, make_vector_shape, make_grid_matrix

FUNC_LOGISTIC = {'l', 'logistic'}
FUNC_WEIBULL = {'w', 'g', 'weibull', 'gumbel'}
FUNC_NORMAL = {'n', 'normal'}

FUNC_VALID = FUNC_LOGISTIC ^ FUNC_WEIBULL ^ FUNC_NORMAL


class Psi(ADOGeneric):
    def __init__(self, func_type, stimulus, guess_rate, lapse_rate, threshold, slope):
        super(Psi, self).__init__()

        assert func_type in FUNC_VALID
        self.func_type = func_type

        self.stimulus = np.array(stimulus).reshape(-1)
        self.guess_rate = np.array(guess_rate).reshape(-1)
        self.lapse_rate = np.array(lapse_rate).reshape(-1)
        self.threshold = np.array(threshold).reshape(-1)
        self.slope = np.array(slope).reshape(-1)

        self.y_obs = np.array([0, 1])  # Binary response

        self.designs = [self.stimulus]
        self.params = [self.guess_rate, self.lapse_rate, self.threshold, self.slope]

        self.label_design = ['stimulus']
        self.label_param = ['guess_rate', 'lapse_rate', 'threshold', 'slope']

        self.grid_design = make_grid_matrix(*self.designs)
        self.grid_param = make_grid_matrix(*self.params)
        self.grid_response = make_grid_matrix(self.y_obs)

        self.idx_opt = get_random_design_index(self.grid_design)
        self.y_obs_prev = 1
        self.d_step = 1

        self.initialize()

    @classmethod
    def compute_p_obs(cls, func_type, stimulus, guess_rate, lapse_rate, threshold, slope):
        r"""
        Calculate the psychometric function given parameters.

        Psychometric functions provide the probability of a subject to recognize
        the stimulus intensity. The base form of the function is as below:

        .. math::
            \Psi(x; \gamma, \lambda, \mu, \beta)
                = \gamma + (1 - \gamma - \lambda) \times F(x; \mu, \beta) \\

        where :math:`x` is the intensity of a given stimulus,
        :math:`\gamma` is the guess rate,
        :math:`\lambda` is the lapse rate,
        :math:`F(x; \mu, \beta)` is a function that defines the shape of a :math:`\Psi` function, and
        :math:`\mu` and :math:`\beta` are the threshold and the slope of the function.

        There are three types of psychometric functions with different :math:`F(x; \mu, \beta)`:

        .. math::
            \begin{align*}
            \text{Logistic function} &\quad
                F(x; \mu, \beta) = \left[
                    1 + \exp\left(-\beta (x - \mu) \right)
                \right]^{-1} \\
            \text{Log Weibull (Gumbel) CDF} &\quad
                F(x; \mu, \beta) = CDF_\text{Gumbel_l}\left( \beta (x - \mu) \right) \\
            \text{Normal CDF} &\quad
                F(x; \mu, \beta) = CDF_\text{Normal}\left( \beta (x - \mu) \right)
            \end{align*}

        Parameters
        ----------
        func_type : {'l', 'w', 'g', 'n'}
            The type of the function used in the Psi function.
            It can have one type among Logistic ('l'), log Weibull ('w' or 'g'),
            and normal CDF ('n').
        stimulus : numpy.ndarray or array_like
        guess_rate : numpy.ndarray or array_like
        lapse_rate : numpy.ndarray or array_like
        threshold : numpy.ndarry or array_like
        slope : numpy.ndarray or array_like

        Returns
        -------
        psi : numpy.ndarray

        """
        assert func_type in FUNC_VALID, 'Invalid func_type is given.'

        f = slope * (stimulus - threshold)
        if func_type in FUNC_LOGISTIC:  # Logistic function
            f = inv_logit(f)
        elif func_type in FUNC_WEIBULL:  # Log Weibull (Gumbel) CDF
            f = gumbel_l.cdf(f)
        elif func_type in FUNC_NORMAL:  # Normal CDF
            f = norm.cdf(f)

        return guess_rate + (1 - guess_rate - lapse_rate) * f

    def _compute_p_obs(self):
        shape_design = make_vector_shape(2, 0)
        shape_param = make_vector_shape(2, 1)

        st = self.grid_design[:, 0].reshape(shape_design)
        gr = self.grid_param[:, 0].reshape(shape_param)
        lr = self.grid_param[:, 1].reshape(shape_param)
        th = self.grid_param[:, 2].reshape(shape_param)
        sl = self.grid_param[:, 3].reshape(shape_param)

        return Psi.compute_p_obs(
            func_type=self.func_type, stimulus=st, guess_rate=gr, lapse_rate=lr, threshold=th, slope=sl)

    def _compute_log_lik(self):
        dim_p_obs = len(self.p_obs.shape)
        y = self.y_obs.reshape(make_vector_shape(dim_p_obs + 1, dim_p_obs))
        p = np.expand_dims(self.p_obs, dim_p_obs)

        return log_lik_bern(y, p)

    def get_design(self, kind='optimal'):
        r"""Choose a design with a given type.

        1. :code:`optimal`: an optimal design :math:`d^*` that maximizes the mutual information.

            .. math::
                \begin{align*}
                    p(y | d) &= \sum_\theta p(y | \theta, d) p_t(\theta) \\
                    I(Y(d); \Theta) &= H(Y(d)) - H(Y(d) | \Theta) \\
                    d^* &= \operatorname*{argmax}_d I(Y(d); |Theta) \\
                \end{align*}

        2. :code:`staircase`: Choose the stimulus :math:`s` as below:

            .. math::
                s_t = \begin{cases}
                    s_{t-1} - 1 & \text{if } y_{t-1} = 1 \\
                    s_{t-1} + 2 & \text{if } y_{t-1} = 0
                \end{cases}

        3. :code:`random`: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'staircase', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : array_like
            A chosen design vector
        """
        assert kind in {'optimal', 'staircase', 'random'}

        self._update_mutual_info()

        def get_design_optimal():
            return self.grid_design[np.argmax(self.mutual_info)]

        def get_design_staircase():
            if self.y_obs_prev == 1:
                idx = max(0, np.array(self.idx_opt)[0] - self.d_step)
            else:
                idx = min(len(self.stimulus) - 1, np.array(self.idx_opt)[0] + self.d_step * 2)

            return self.grid_design[np.int(idx)]

        def get_design_random():
            return self.grid_design[get_random_design_index(self.grid_design)]

        if kind == 'optimal':
            ret = get_design_optimal()
        elif kind == 'staircase':
            ret = get_design_staircase()
        elif kind == 'random':
            ret = get_design_random()
        else:
            raise RuntimeError('An invalid kind of design: "{}".'.format(type))

        self.idx_opt = get_nearest_grid_index(ret, self.grid_design)

        return ret

    def update(self, design, response, store=True):
        super(Psi, self).update(design, response, store)

        self.y_obs_prev = response
