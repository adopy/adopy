from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from adopy.functions import (
    get_random_design_index,
    make_grid_matrix,
    marginalize,
    make_vector_shape,
    logsumexp,
)
from adopy.types import array_like, vector_like, matrix_like
from adopy.cmodules import get_nearest_grid_index

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
                 grid_response: Dict[str, Any],
                 noise_ratio: Optional[float] = 1e-7,
                 dtype: Optional[Any] = np.float32):
        super(Engine, self).__init__()

        if model.task != task:
            raise ValueError('Given task and model are not matched.')

        self._task = task
        self._model = model
        self._dtype = dtype
        self._noise_ratio = noise_ratio

        self._g_d = g_d = make_grid_matrix(grid_design)[task.designs]
        self._g_p = g_p = make_grid_matrix(grid_param)[model.params]
        self._g_y = g_y = make_grid_matrix(grid_response)[task.responses]

        self.n_d = g_d.shape[0]
        self.n_p = g_p.shape[0]
        self.n_y = g_y.shape[0]

        l_model = np.exp(self._compute_log_lik())
        l_noise = np.repeat(1 / self.n_y, self.n_y).reshape(1, 1, -1)
        log_lik = np.log((1 - noise_ratio) * l_model + noise_ratio * l_noise) \
            .astype(dtype)

        self.log_lik = log_lik
        self.ent_obs = -np.einsum('ijk,ijk->ij', np.exp(log_lik), log_lik)
        self.log_prior = np.repeat(np.log(1 / self.n_p), self.n_p) \
            .astype(dtype)
        self.log_post = np.repeat(np.log(1 / self.n_p), self.n_p) \
            .astype(dtype)

        self.mll = np.sum(
            self.log_lik + self.log_post.reshape(1, -1, 1), axis=1)
        ent_marg = -np.einsum('ij,ij->i', np.exp(self.mll), self.mll)
        ent_cond = np.einsum('j,ij->i', np.exp(self.log_post), self.ent_obs)
        self.mutual_info = ent_marg - ent_cond

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
    def grid_design(self):
        """Grids for a design space."""
        return self._g_d

    @property
    def grid_param(self):
        """Grids for a parameter space."""
        return self._g_p

    @property
    def grid_response(self):
        """Grids for a response space."""
        return self._g_y

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
    def dtype(self):
        return self._dtype

    ###########################################################################
    # Methods
    ###########################################################################

    def _compute_log_lik(self):
        """Compute the probability of getting observed response."""
        shape_design = make_vector_shape(3, 0)
        shape_param = make_vector_shape(3, 1)
        shape_response = make_vector_shape(3, 2)

        args = {}
        args.update({
            k: v.reshape(shape_design)
            for k, v in self.task.extract_designs(self.grid_design).items()
        })
        args.update({
            k: v.reshape(shape_param)
            for k, v in self.model.extract_params(self.grid_param).items()
        })
        args.update({
            k: v.reshape(shape_response)
            for k, v in self.task.extract_responses(self.grid_response).items()
        })

        return self.model.compute(**args)

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self.log_post = np.repeat(np.log(1 / self.n_p), self.n_p) \
            .astype(self.dtype)
        self._update_mutual_info()

    def _update_mutual_info(self):
        """
        Update mutual information using posterior distributions.

        If there is no nedd to update mutual information, it ends.
        The flag to update mutual information must be true only when
        posteriors are updated in :code:`update(design, response)` method.
        """
        self.mll = np.sum(self.log_lik + self.log_post.reshape(1, self.n_p, 1),
                          axis=1)
        ent_marg = -np.einsum('ij,ij->i', np.exp(self.mll), self.mll)
        ent_cond = np.einsum('j,ij->i', np.exp(self.log_post), self.ent_obs)
        self.mutual_info = ent_marg - ent_cond

    def get_design(self, kind='optimal') -> Dict[str, Any]:
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
        design : Dict[str, any]
            A chosen design vector
        """

        if kind == 'optimal':
            idx_design = np.argmax(self.mutual_info)

        elif kind == 'random':
            idx_design = get_random_design_index(self.grid_design)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')

        return self.grid_design.iloc[idx_design].to_dict()

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
            design = pd.Series(
                design, index=self.task.designs, dtype=np.float32)

        if not isinstance(response, pd.Series):
            response = pd.Series(
                response, index=self.task.responses, dtype=np.float32)

        i_d = get_nearest_grid_index(design.values, self.grid_design.values)
        i_y = get_nearest_grid_index(
            response.values, self.grid_response.values)

        self.log_post += self.log_lik[i_d, :, i_y]
        self.log_post -= logsumexp(self.log_post)
        self._update_mutual_info()
