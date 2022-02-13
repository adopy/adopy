# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Callable

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from adopy.functions import make_grid_matrix
from adopy.types import array_like, matrix_like, vector_like

from ._model import ModelV2
from ._task import TaskV2

__all__ = ["JaxEngineV1", "JaxEngineV2"]


@jax.jit
def get_marg_log_lik(ll, lp):
    return logsumexp(ll + lp.reshape(1, -1, 1), axis=1)


@jax.jit
def get_ent(ll):
    return -1 * jnp.einsum("dpy,dpy->dp", jnp.exp(ll), ll)


@jax.jit
def get_ent_marg(mll):
    return -1 * jnp.einsum("dy,dy->d", jnp.exp(mll), mll)


@jax.jit
def get_ent_cond(post, ent):
    return jnp.einsum("p,dp->d", post, ent)


@jax.jit
def get_nearest_grid_index(design, designs) -> int:
    ds = designs
    d = design.reshape(1, -1)
    return jnp.square(ds - d).sum(-1).argsort()[0]


@jax.jit
def update_log_post(i_d, i_y, log_post, log_lik):
    ret = log_post + log_lik[i_d, :, i_y]
    return ret - logsumexp(ret)


class JaxEngineV1(object):
    def __init__(
        self,
        designs: List[str],
        parameters: List[str],
        responses: List[str],
        model_func: Callable,
        grid_design: Dict[str, Any],
        grid_param: Dict[str, Any],
        grid_response: Dict[str, Any],
        noise_ratio: float = 1e-7,
        dtype: Optional[Any] = jnp.float32,
    ):
        self.designs = designs
        self.parameters = parameters
        self.responses = responses

        self.model_func = model_func
        self.noise_ratio = noise_ratio
        self.dtype = dtype

        self._g_d = jnp.array(make_grid_matrix(grid_design)[designs].values)
        self._g_p = jnp.array(make_grid_matrix(grid_param)[parameters].values)
        self._g_y = jnp.array(make_grid_matrix(grid_response)[responses].values)

        self.n_d = self._g_d.shape[0]
        self.n_p = self._g_p.shape[0]
        self.n_y = self._g_y.shape[0]

        self._log_lik = None
        self._marg_log_lik = None
        self._ent = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self.log_prior = jnp.log(jnp.ones(self.n_p, dtype=dtype) / self.n_p)
        self.log_post = None

        self.reset()

    @property
    def grid_design(self):
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.
        """
        return pd.DataFrame(self._g_d, columns=self.designs)

    @property
    def grid_param(self):
        """
        Grid space for model parameters, generated from the grid definition,
        given as :code:`grid_param` with initialization.
        """
        return pd.DataFrame(self._g_p, columns=self.parameters)

    @property
    def grid_response(self):
        """
        Grid space for response variables, generated from the grid definition,
        given as :code:`grid_response` with initialization.
        """
        return pd.DataFrame(self._g_y, columns=self.responses)

    @property
    def prior(self) -> vector_like:
        r"""
        Prior probabilities on the grid space of model parameters, :math:`p_0(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return jnp.exp(self.log_prior)

    @property
    def post(self) -> vector_like:
        r"""
        Posterior probabilities on the grid space of model parameters, :math:`p(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return jnp.exp(self.log_post)

    @property
    def log_lik(self) -> matrix_like:
        r"""
        Log likelihood :math:`p(y | d, \theta)` for all discretized values of
        :math:`y`, :math:`d`, and :math:`\theta`.
        """
        if self._log_lik is None:
            shape_design = (-1, 1, 1)
            shape_param = (1, -1, 1)
            shape_response = (1, 1, -1)

            args = {}
            args.update(
                {
                    k: self._g_d[:, i].reshape(shape_design)
                    for i, k in enumerate(self.designs)
                }
            )
            args.update(
                {
                    k: self._g_p[:, i].reshape(shape_param)
                    for i, k in enumerate(self.parameters)
                }
            )
            args.update(
                {
                    k: self._g_y[:, i].reshape(shape_response)
                    for i, k in enumerate(self.responses)
                }
            )

            l_model = jnp.exp(self.model_func(**args))

            self._log_lik = jnp.log(
                (1 - 2 * self.noise_ratio) * l_model + self.noise_ratio
            ).astype(self.dtype)

        return self._log_lik

    @property
    def marg_log_lik(self) -> array_like:
        r"""
        Marginal log likelihood :math:`\log p(y | d)` for all discretized values
        for :math:`y` and :math:`d`.
        """
        if self._marg_log_lik is None:
            self._marg_log_lik = logsumexp(
                self.log_lik + self.log_post.reshape(1, -1, 1), axis=1
            )
        return self._marg_log_lik

    @property
    def ent(self) -> array_like:
        r"""
        Entropy :math:`H(Y(d) | \theta) = -\sum_y p(y | d, \theta) \log p(y | d, \theta)`
        for all discretized values for :math:`d` and :math:`\theta`.
        """
        if self._ent is None:
            self._ent = -1 * jnp.einsum(
                "dpy,dpy->dp", jnp.exp(self.log_lik), self.log_lik
            )

        return self._ent

    @property
    def ent_marg(self) -> array_like:
        r"""
        Marginal entropy :math:`H(Y(d)) = -\sum_y p(y | d) \log p(y | d)`
        for all discretized values for :math:`d`,
        where :math:`p(y | d)` indicates the marginal likelihood.
        """
        if self._ent_marg is None:
            self._ent_marg = -1 * jnp.einsum(
                "dy,dy->d", jnp.exp(self.marg_log_lik), self.marg_log_lik
            )
        return self._ent_marg

    @property
    def ent_cond(self) -> array_like:
        r"""
        Conditional entropy :math:`H(Y(d) | \Theta) = \sum_\theta p(\theta) H(Y(d) | \theta)`
        for all discretized values for :math:`d`,
        where :math:`p(\theta)` indicates the posterior distribution for model parameters.
        """
        if self._ent_cond is None:
            self._ent_cond = jnp.einsum("p,dp->d", self.post, self.ent)
        return self._ent_cond

    @property
    def mutual_info(self) -> vector_like:
        r"""
        Mutual information :math:`I(Y(d); \Theta) = H(Y(d)) - H(Y(d) | \Theta)`,
        where :math:`H(Y(d))` indicates the marginal entropy
        and :math:`H(Y(d) | \Theta)` indicates the conditional entropy.
        """
        if self._mutual_info is None:
            self._mutual_info = self.ent_marg - self.ent_cond
        return self._mutual_info

    def _update_mutual_info(self):
        """
        Update mutual information using posterior distributions.

        By accessing :code:`mutual_info` once, the engine computes log_lik,
        marg_log_lik, ent, ent_marg, ent_cond, and mutual_info in a chain.
        """
        _ = self.mutual_info

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self._log_lik = None
        self._marg_log_lik = None
        self._ent = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self.log_post = self.log_prior.copy()

        self._update_mutual_info()

    def get_design(self, kind="optimal") -> Optional[Dict[str, Any]]:
        r"""
        Choose a design with given one of following types:

        * :code:`'optimal'` (default): an optimal design :math:`d^*` that maximizes the mutual
          information.
        * :code:`'random'`: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose. Default is :code:`'optimal'`.

        Returns
        -------
        design : Dict[str, any] or None
            A chosen design vector to use for the next trial.
            Returns `None` if there is no design available.
        """
        if len(self.designs) == 0:
            return None

        if kind == "optimal":
            idx_design = jnp.argmax(self.mutual_info)

        elif kind == "random":
            idx_design = np.random.randint(self.n_d)

        else:
            raise ValueError('The argument kind should be "optimal" or "random".')

        return self._g_d[idx_design, :]

    def update(self, design, response):
        r"""
        Update the posterior probabilities :math:`p(\theta | y, d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y, d^*) \sim
                p( y | \theta, d^*) p(\theta)

        .. code-block:: python

            # Given design and resposne as `design` and `response`,
            # the engine can update probability with the following line:
            engine.update(design, response)

        Also, it can takes multiple observations for updating posterior
        probabilities. Multiple pairs of design and response should be
        given as a list of designs and a list of responses, into
        :code:`design` and :code:`response` argument, respectively.

        .. math::

            \begin{aligned}
            p\big(\theta | y_1, \ldots, y_n, d_1^*, \ldots, d_n^*\big)
            &\sim p\big( y_1, \ldots, y_n | \theta, d_1^*, \ldots, d_n^* \big) p(\theta) \\
            &= p(y_1 | \theta, d_1^*) \cdot \ldots \cdot p(y_n | \theta, d_n^*) p(\theta)
            \end{aligned}

        .. code-block:: python

            # Given a list of designs and corresponding responses as below:
            designs = [design1, design2, design3]
            responses = [response1, response2, response3]

            # the engine can update with multiple observations:
            engine.update(designs, responses)

        Parameters
        ----------
        design : dict or :code:`pandas.Series` or list of designs
            Design vector for given response
        response : dict or :code:`pandas.Series` or list of responses
            Any kinds of observed response
        """
        d = jnp.array(design, dtype=self.dtype)
        y = jnp.array(response, dtype=self.dtype)

        i_d = get_nearest_grid_index(d, self._g_d)
        i_y = get_nearest_grid_index(y, self._g_y)
        self.log_post = update_log_post(i_d, i_y, self.log_post, self.log_lik)

        self._marg_log_lik = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self._update_mutual_info()


class JaxEngineV2(object):
    """
    A base class for an ADO engine to compute optimal designs.
    """

    def __init__(
        self,
        *,
        task: TaskV2,
        model: ModelV2,
        noise_ratio: float = 1e-7,
        dtype: Optional[Any] = np.float32
    ):
        super(JaxEngineV2, self).__init__()

        if model.task != task:
            raise ValueError("Given task and model are not matched.")

        self._task = task
        self._model = model
        self._dtype = dtype
        self._noise_ratio = noise_ratio

        self._g_d = self.task.get_grid_design_jax()
        self._g_p = self.model.get_grid_param_jax()
        self._g_y = self.task.get_grid_response_jax()

        self.n_d = self._g_d.shape[0]
        self.n_p = self._g_p.shape[0]
        self.n_y = self._g_y.shape[0]

        self._log_lik = None
        self._marg_log_lik = None
        self._ent = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self.log_prior = jnp.log(jnp.ones(self.n_p, dtype=dtype) / self.n_p)
        self.log_post = None

        self.reset()

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def task(self) -> TaskV2:
        """Task instance for the engine."""
        return self._task

    @property
    def model(self) -> ModelV2:
        """Model instance for the engine."""
        return self._model

    @property
    def grid_design(self) -> pd.DataFrame:
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.
        """
        return self.task.grid_design

    @property
    def grid_param(self) -> pd.DataFrame:
        """
        Grid space for model parameters, generated from the grid definition,
        given as :code:`grid_param` with initialization.
        """
        return self.model.grid_param

    @property
    def grid_response(self) -> pd.DataFrame:
        """
        Grid space for response variables, generated from the grid definition,
        given as :code:`grid_response` with initialization.
        """
        return self.task.grid_response

    @property
    def log_prior(self) -> vector_like:
        r"""
        Log prior probabilities on the grid space of model parameters,
        :math:`\log p_0(\theta)`. This log probabilities correspond to grid
        points defined in :code:`grid_param`.
        """
        return self._log_prior

    @log_prior.setter
    def log_prior(self, lp):
        self._log_prior = lp

    @log_prior.deleter
    def log_prior(self):
        del self._log_prior
        self._log_prior = np.log(np.ones(self.n_p) / self.n_p, dtype=self.dtype)

    @property
    def log_post(self) -> vector_like:
        r"""
        Log posterior probabilities on the grid space of model parameters,
        :math:`\log p(\theta)`. This log probabilities correspond to grid
        points defined in :code:`grid_param`.
        """
        return self._log_post

    @log_post.setter
    def log_post(self, lp):
        self._log_post = lp

    @log_post.deleter
    def log_post(self):
        del self._log_post
        self._log_post = np.copy(self._log_prior)

    @property
    def prior(self) -> vector_like:
        r"""
        Prior probabilities on the grid space of model parameters, :math:`p_0(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return np.exp(self._log_prior)

    @property
    def post(self) -> vector_like:
        r"""
        Posterior probabilities on the grid space of model parameters, :math:`p(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return np.exp(self._log_post)

    # @property
    # def marg_post(self) -> Dict[str, vector_like]:
    #     """Marginal posterior distributions for each parameter"""
    #     return {
    #         param: marginalize(self.post, self.grid_param, i)
    #         for i, param in enumerate(self.model.params)
    #     }

    @property
    def log_lik(self) -> matrix_like:
        r"""
        Log likelihood :math:`p(y | d, \theta)` for all discretized values of
        :math:`y`, :math:`d`, and :math:`\theta`.
        """
        if self._log_lik is None:
            shape_design = (-1, 1, 1)
            shape_param = (1, -1, 1)
            shape_response = (1, 1, -1)

            args = {}
            args.update(
                {
                    k: self._g_d[:, i].reshape(shape_design)
                    for i, k in enumerate(self.task.designs)
                }
            )
            args.update(
                {
                    k: self._g_p[:, i].reshape(shape_param)
                    for i, k in enumerate(self.model.params)
                }
            )
            args.update(
                {
                    k: self._g_y[:, i].reshape(shape_response)
                    for i, k in enumerate(self.task.responses)
                }
            )

            l_model = jnp.exp(self.model.compute(**args))

            self._log_lik = jnp.log(
                (1 - 2 * self._noise_ratio) * l_model + self._noise_ratio
            ).astype(self.dtype)

        return self._log_lik

    @property
    def marg_log_lik(self) -> array_like:
        r"""
        Marginal log likelihood :math:`\log p(y | d)` for all discretized values
        for :math:`y` and :math:`d`.
        """
        if self._marg_log_lik is None:
            self._marg_log_lik = get_marg_log_lik(self.log_lik, self.log_post)
        return self._marg_log_lik

    @property
    def ent(self) -> array_like:
        r"""
        Entropy :math:`H(Y(d) | \theta) = -\sum_y p(y | d, \theta) \log p(y | d, \theta)`
        for all discretized values for :math:`d` and :math:`\theta`.
        """
        if self._ent is None:
            self._ent = get_ent(self.log_lik)
        return self._ent

    @property
    def ent_marg(self) -> array_like:
        r"""
        Marginal entropy :math:`H(Y(d)) = -\sum_y p(y | d) \log p(y | d)`
        for all discretized values for :math:`d`,
        where :math:`p(y | d)` indicates the marginal likelihood.
        """
        if self._ent_marg is None:
            self._ent_marg = get_ent_marg(self.marg_log_lik)
        return self._ent_marg

    @property
    def ent_cond(self) -> array_like:
        r"""
        Conditional entropy :math:`H(Y(d) | \Theta) = \sum_\theta p(\theta) H(Y(d) | \theta)`
        for all discretized values for :math:`d`,
        where :math:`p(\theta)` indicates the posterior distribution for model parameters.
        """
        if self._ent_cond is None:
            self._ent_cond = get_ent_cond(self.post, self.ent)
        return self._ent_cond

    @property
    def mutual_info(self) -> vector_like:
        r"""
        Mutual information :math:`I(Y(d); \Theta) = H(Y(d)) - H(Y(d) | \Theta)`,
        where :math:`H(Y(d))` indicates the marginal entropy
        and :math:`H(Y(d) | \Theta)` indicates the conditional entropy.
        """
        if self._mutual_info is None:
            self._mutual_info = self.ent_marg - self.ent_cond
        return self._mutual_info

    @property
    def post_mean(self) -> vector_like:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``num_params``.
        """
        return pd.Series(
            np.dot(self.post, self.grid_param),
            index=self.model.params,
            name="Posterior mean",
        )

    @property
    def post_cov(self) -> np.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, num_params)``.
        """
        # shape: (N_grids, N_param)
        d = self.grid_param.values - self.post_mean.values
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self) -> vector_like:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``num_params``.
        """
        return pd.Series(
            np.sqrt(np.diag(self.post_cov)),
            index=self.model.params,
            name="Posterior SD",
        )

    @property
    def dtype(self):
        """
        The desired data-type for the internal vectors and matrixes, e.g.,
        :code:`numpy.float64`. Default is :code:`numpy.float32`.

        .. versionadded:: 0.4.0
        """
        return self._dtype

    ###########################################################################
    # Methods
    ###########################################################################

    def _update_mutual_info(self):
        """
        Update mutual information using posterior distributions.

        By accessing :code:`mutual_info` once, the engine computes log_lik,
        marg_log_lik, ent, ent_marg, ent_cond, and mutual_info in a chain.
        """
        _ = self.mutual_info

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self._log_lik = None
        self._marg_log_lik = None
        self._ent = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self.log_post = self.log_prior.copy()

        self._update_mutual_info()

    def get_design(self, kind="optimal") -> Optional[Dict[str, Any]]:
        r"""
        Choose a design with given one of following types:

        * :code:`'optimal'` (default): an optimal design :math:`d^*` that maximizes the mutual
          information.
        * :code:`'random'`: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose. Default is :code:`'optimal'`.

        Returns
        -------
        design : Dict[str, any] or None
            A chosen design vector to use for the next trial.
            Returns `None` if there is no design available.
        """
        if len(self.task.designs) == 0:
            return None

        if kind == "optimal":
            idx_design = np.argmax(self.mutual_info)

        elif kind == "random":
            idx_design = np.random.randint(self.n_d)

        else:
            raise ValueError('The argument kind should be "optimal" or "random".')

        return self.grid_design.iloc[idx_design].to_dict()

    def update(self, design, response):
        r"""
        Update the posterior probabilities :math:`p(\theta | y, d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y, d^*) \sim
                p( y | \theta, d^*) p(\theta)

        .. code-block:: python

            # Given design and resposne as `design` and `response`,
            # the engine can update probability with the following line:
            engine.update(design, response)

        Also, it can takes multiple observations for updating posterior
        probabilities. Multiple pairs of design and response should be
        given as a list of designs and a list of responses, into
        :code:`design` and :code:`response` argument, respectively.

        .. math::

            \begin{aligned}
            p\big(\theta | y_1, \ldots, y_n, d_1^*, \ldots, d_n^*\big)
            &\sim p\big( y_1, \ldots, y_n | \theta, d_1^*, \ldots, d_n^* \big) p(\theta) \\
            &= p(y_1 | \theta, d_1^*) \cdot \ldots \cdot p(y_n | \theta, d_n^*) p(\theta)
            \end{aligned}

        .. code-block:: python

            # Given a list of designs and corresponding responses as below:
            designs = [design1, design2, design3]
            responses = [response1, response2, response3]

            # the engine can update with multiple observations:
            engine.update(designs, responses)

        Parameters
        ----------
        design : dict or :code:`pandas.Series` or list of designs
            Design vector for given response
        response : dict or :code:`pandas.Series` or list of responses
            Any kinds of observed response
        """
        d = jnp.array(design, dtype=self.dtype)
        y = jnp.array(response, dtype=self.dtype)

        i_d = get_nearest_grid_index(d, self._g_d)
        i_y = get_nearest_grid_index(y, self._g_y)
        self.log_post = update_log_post(i_d, i_y, self.log_post, self.log_lik)

        self._marg_log_lik = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self._update_mutual_info()
