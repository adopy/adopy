from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from adopy.functions import (
    expand_multiple_dims,
    get_nearest_grid_index,
    get_random_design_index,
    make_grid_matrix,
    marginalize,
    make_vector_shape,
    log_lik_bernoulli
)
from adopy.types import array_like, vector_like, matrix_like

from ._task import Task
from ._model import Model

__all__ = ['Engine']


class Engine(object):
    """
    A base class for an ADO engine to compute optimal designs.
    """

    def __init__(self,
                 task: Task,
                 model: Model,
                 grid_design: Dict[str, Any],
                 grid_param: Dict[str, Any],
                 lambda_et: Optional[float] = None):
        super(Engine, self).__init__()

        if model.task != task:
            raise ValueError('Given task and model are not matched.')

        self._task = task  # type: Task
        self._model = model  # type: Model
        self.lambda_et = lambda_et

        self.grid_design = make_grid_matrix(grid_design)[task.designs]
        self.grid_param = make_grid_matrix(grid_param)[model.params]
        self.grid_response = pd.DataFrame(
            np.array(task.responses), columns=['y_obs'])

        self.reset()

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def task(self) -> Task:
        """Task instance for the engine"""
        return self._task

    @property
    def model(self) -> Model:
        """Model instance for the engine"""
        return self._model

    @property
    def num_design(self):
        """Number of design grid axes"""
        return len(self.task.designs)

    @property
    def num_param(self):
        """Number of parameter grid axes"""
        return len(self.model.params)

    @property
    def prior(self) -> array_like:
        """Prior distributions of joint parameter space"""
        return np.exp(self.log_prior)

    @property
    def post(self) -> array_like:
        """Posterior distributions of joint parameter space"""
        return np.exp(self.log_post)

    @property
    def marg_post(self) -> Dict[str, vector_like]:
        """Marginal posterior distributions for each parameter"""
        return {
            param: marginalize(self.post, self.grid_param, i)
            for i, param in enumerate(self.model.params)
        }

    @property
    def post_mean(self) -> vector_like:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``num_params``.
        """
        return np.dot(self.post, self.grid_param)

    @property
    def post_cov(self) -> np.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, num_params)``.
        """
        # shape: (N_grids, N_param)
        d = self.grid_param.values - self.post_mean
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self) -> vector_like:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``num_params``.
        """
        return np.sqrt(np.diag(self.post_cov))

    @property
    def lambda_et(self):
        """
        Lambda value for eligibility traces. If it equals to None, eligibility
        traces do not affect choices of the optimal designs.
        """
        return self._lambda_et

    @lambda_et.setter
    def lambda_et(self, v):
        if v and not (0 <= v <= 1):
            raise ValueError('Invalid value for lambda_et')

        self._lambda_et = v

    ###########################################################################
    # Methods
    ###########################################################################

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self.y_obs = np.array(self.task.responses)
        self.p_obs = self._compute_p_obs()
        self.log_lik = ll = self._compute_log_lik()

        lp = np.ones(self.grid_param.shape[0])
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

        lp = expand_multiple_dims(self.log_post, 1, 1)
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

        self.ent_obs = -np.multiply(np.exp(ll), ll).sum(-1)
        self.ent_marg = None
        self.ent_cond = None
        self.mutual_info = None

        self.eligibility_trace = np.zeros(self.grid_design.shape[0])

        self.flag_update_mutual_info = True

    def _compute_p_obs(self):
        """Compute the probability of getting observed response."""
        shape_design = make_vector_shape(2, 0)
        shape_param = make_vector_shape(2, 1)

        args = {}
        args.update({
            k: v.reshape(shape_design)
            for k, v in self.task.extract_designs(self.grid_design).items()
        })
        args.update({
            k: v.reshape(shape_param)
            for k, v in self.model.extract_params(self.grid_param).items()
        })

        return self.model.compute(**args)

    def _compute_log_lik(self):
        """Compute the log likelihood."""
        dim_p_obs = len(self.p_obs.shape)
        y = self.y_obs.reshape(make_vector_shape(dim_p_obs + 1, dim_p_obs))
        p = np.expand_dims(self.p_obs, dim_p_obs)

        return log_lik_bernoulli(y, p)

    def _update_mutual_info(self):
        """
        Update mutual information using posterior distributions.

        If there is no nedd to update mutual information, it ends.
        The flag to update mutual information must be true only when
        posteriors are updated in :code:`update(design, response)` method.
        """
        # If there is no need to update mutual information, it ends.
        if not self.flag_update_mutual_info:
            return

        # Calculate the marginal log likelihood.
        lp = expand_multiple_dims(self.log_post, 1, 1)
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

        # Calculate the marginal entropy and conditional entropy.
        self.ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        self.ent_cond = np.sum(
            self.post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - \
            self.ent_cond  # shape (num_designs,)

        # Flag that there is no need to update mutual information again.
        self.flag_update_mutual_info = False

    def get_design(self, kind='optimal'):
        # type: (str) -> pd.Series
        r"""
        Choose a design with a given type.

        * ``optimal``: an optimal design :math:`d^*` that maximizes the mutual
          information.
        * ``random``: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : array_like
            A chosen design vector
        """

        if kind == 'optimal':
            self._update_mutual_info()
            idx_design = np.argmax(
                self.mutual_info * (1 - self.eligibility_trace))

        elif kind == 'random':
            idx_design = get_random_design_index(self.grid_design)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')

        return self.grid_design.iloc[idx_design]

    def update(self, design, response):
        r"""
        Update the posterior :math:`p(\theta | y_\text{obs}(t), d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y_\text{obs}(t), d^*) =
                \frac{ p( y_\text{obs}(t) | \theta, d^*) p_t(\theta) }
                    { p( y_\text{obs}(t) | d^* ) }

        Parameters
        ----------
        design
            Design vector for given response
        response
            Any kinds of observed response
        """
        if not isinstance(design, pd.Series):
            design = pd.Series(design, index=self.task.designs)

        idx_design = get_nearest_grid_index(design, self.grid_design)
        idx_response = get_nearest_grid_index(
            pd.Series(response), self.grid_response)

        self.log_post += self.log_lik[idx_design, :, idx_response].flatten()
        self.log_post -= logsumexp(self.log_post)

        if self.lambda_et:
            self.eligibility_trace *= self.lambda_et
            self.eligibility_trace[idx_design] += 1

        self.flag_update_mutual_info = True
