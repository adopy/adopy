from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.special import logsumexp

from adopy.functions import (
    expand_multiple_dims,
    get_nearest_grid_index,
    get_random_design_index,
    make_grid_matrix,
    marginalize,
    make_vector_shape,
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
                 grid_response: Dict[str, Any],
                 dtype: Optional[Any] = np.float32):
        super(Engine, self).__init__()

        if model.task != task:
            raise ValueError('Given task and model are not matched.')

        self._task = task  # type: Task
        self._model = model  # type: Model
        self._dtype = dtype

        self._grid_design = make_grid_matrix(grid_design)[task.designs]
        self._grid_param = make_grid_matrix(grid_param)[model.params]
        self._grid_response = make_grid_matrix(grid_response)[task.responses]

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
    def grid_design(self):
        """Grids for a design space."""
        return self._grid_design

    @property
    def grid_param(self):
        """Grids for a parameter space."""
        return self._grid_param

    @property
    def grid_response(self):
        """Grids for a response space."""
        return self._grid_response

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
        """
        Datatype for internal grid objects.
        """
        return self._dtype

    ###########################################################################
    # Methods
    ###########################################################################

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self.log_lik = ll = self._compute_log_lik()

        lp = np.ones(self.grid_param.shape[0], dtype=self.dtype)
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

        lp = expand_multiple_dims(self.log_post, 1, 1)
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

        self.ent_obs = -np.multiply(np.exp(ll), ll, dtype=self.dtype).sum(-1)
        self.ent_marg = None
        self.ent_cond = None
        self.mutual_info = None

        self.flag_update_mutual_info = True

    def _compute_log_lik(self):
        """Compute the probability of getting observed response."""
        shape_design = make_vector_shape(3, 0)
        shape_param = make_vector_shape(3, 1)
        shape_response = make_vector_shape(3, 2)

        args = {}
        args.update({
            k: v.reshape(shape_design).astype(self.dtype)
            for k, v in self.task.extract_designs(self.grid_design).items()
        })
        args.update({
            k: v.reshape(shape_param).astype(self.dtype)
            for k, v in self.model.extract_params(self.grid_param).items()
        })
        args.update({
            k: v.reshape(shape_response)
            for k, v in self.task.extract_responses(self.grid_response).items()
        })

        return self.model.compute(**args)

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
        self.ent_marg = \
            -np.sum(np.exp(mll) * mll, -1)  # shape: (num_designs,)
        self.ent_cond = \
            np.sum(self.post * self.ent_obs, axis=1)  # shape: (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = \
            self.ent_marg - self.ent_cond  # shape: (num_designs,)

        # Flag that there is no need to update mutual information again.
        self.flag_update_mutual_info = False

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
            self._update_mutual_info()
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
            design = pd.Series(design, index=self.task.designs)

        if not isinstance(response, pd.Series):
            response = pd.Series(response, index=self.task.responses)

        idx_design = get_nearest_grid_index(design, self.grid_design)
        idx_response = get_nearest_grid_index(response, self.grid_response)

        self.log_post += self.log_lik[idx_design, :, idx_response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self.flag_update_mutual_info = True
