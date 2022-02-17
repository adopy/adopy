# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from .grid import GridSpace
from .model import Model
from .task import Task

__all__ = ["Engine"]

MK = TypeVar("MK", float, Tuple[float])


def marginalize(
    post: jnp.ndarray, grid: jnp.ndarray, axis: Union[int, List[int]]
) -> Dict[MK, float]:
    """Return marginal distributions from grid-shaped posteriors"""
    assert len(post) == len(grid)

    ret = {}
    for v, p in zip(grid[:, axis], post):
        vv = v.tolist()
        k = vv if jnp.isscalar(vv) else tuple(vv)
        ret[k] = ret.get(k, 0) + p.tolist()

    return dict(sorted(ret.items(), key=lambda x: x[0]))


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
def update_log_post(i_d, i_y, log_post, log_lik):
    ret = log_post + log_lik[i_d, :, i_y]
    return ret - logsumexp(ret)


class Engine(object):
    """
    A base class for an ADO engine to compute optimal designs.
    """

    def __init__(
        self,
        *,
        task: Task,
        model: Model,
        noise_ratio: float = 1e-7,
        dtype: Optional[Any] = np.float32
    ):
        super().__init__()

        if model.task != task:
            raise ValueError("Given task and model are not matched.")

        self._task = task
        self._model = model
        self._dtype = dtype
        self._noise_ratio = noise_ratio

        self._g_d = self.task.grid_design.value
        self._g_p = self.model.grid_param.value
        self._g_y = self.task.grid_response.value

        self.n_d = len(self._g_d)
        self.n_p = len(self._g_p)
        self.n_y = len(self._g_y)

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
    def task(self) -> Task:
        """Task instance for the engine."""
        return self._task

    @property
    def model(self) -> Model:
        """Model instance for the engine."""
        return self._model

    @property
    def grid_design(self) -> GridSpace:
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.
        """
        return self.task.grid_design

    @property
    def grid_param(self) -> GridSpace:
        """
        Grid space for model parameters, generated from the grid definition,
        given as :code:`grid_param` with initialization.
        """
        return self.model.grid_param

    @property
    def grid_response(self) -> GridSpace:
        """
        Grid space for response variables, generated from the grid definition,
        given as :code:`grid_response` with initialization.
        """
        return self.task.grid_response

    @property
    def log_prior(self) -> jnp.ndarray:
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
        self._log_prior = jnp.log(jnp.ones(self.n_p) / self.n_p, dtype=self.dtype)

    @property
    def log_post(self) -> jnp.ndarray:
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
        self._log_post = jnp.zeros_like(self._log_prior)
        self._log_post += self._log_prior

    @property
    def prior(self) -> jnp.ndarray:
        r"""
        Prior probabilities on the grid space of model parameters, :math:`p_0(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return jnp.exp(self._log_prior)

    @property
    def post(self) -> jnp.ndarray:
        r"""
        Posterior probabilities on the grid space of model parameters, :math:`p(\theta)`.
        This probabilities correspond to grid points defined in :code:`grid_param`.
        """
        return jnp.exp(self._log_post)

    @property
    def marg_post(self) -> Dict[str, jnp.array]:
        """Marginal posterior distributions for each parameter"""
        return {
            param: marginalize(self.post, self.grid_param.value, i)
            for i, param in enumerate(self.model.params)
        }

    @property
    def log_lik(self) -> jnp.ndarray:
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
    def marg_log_lik(self) -> jnp.ndarray:
        r"""
        Marginal log likelihood :math:`\log p(y | d)` for all discretized values
        for :math:`y` and :math:`d`.
        """
        if self._marg_log_lik is None:
            self._marg_log_lik = get_marg_log_lik(self.log_lik, self.log_post)
        return self._marg_log_lik

    @property
    def ent(self) -> jnp.ndarray:
        r"""
        Entropy :math:`H(Y(d) | \theta) = -\sum_y p(y | d, \theta) \log p(y | d, \theta)`
        for all discretized values for :math:`d` and :math:`\theta`.
        """
        if self._ent is None:
            self._ent = get_ent(self.log_lik)
        return self._ent

    @property
    def ent_marg(self) -> jnp.ndarray:
        r"""
        Marginal entropy :math:`H(Y(d)) = -\sum_y p(y | d) \log p(y | d)`
        for all discretized values for :math:`d`,
        where :math:`p(y | d)` indicates the marginal likelihood.
        """
        if self._ent_marg is None:
            self._ent_marg = get_ent_marg(self.marg_log_lik)
        return self._ent_marg

    @property
    def ent_cond(self) -> jnp.ndarray:
        r"""
        Conditional entropy :math:`H(Y(d) | \Theta) = \sum_\theta p(\theta) H(Y(d) | \theta)`
        for all discretized values for :math:`d`,
        where :math:`p(\theta)` indicates the posterior distribution for model parameters.
        """
        if self._ent_cond is None:
            self._ent_cond = get_ent_cond(self.post, self.ent)
        return self._ent_cond

    @property
    def mutual_info(self) -> jnp.ndarray:
        r"""
        Mutual information :math:`I(Y(d); \Theta) = H(Y(d)) - H(Y(d) | \Theta)`,
        where :math:`H(Y(d))` indicates the marginal entropy
        and :math:`H(Y(d) | \Theta)` indicates the conditional entropy.
        """
        if self._mutual_info is None:
            self._mutual_info = self.ent_marg - self.ent_cond
        return self._mutual_info

    @property
    def post_mean(self) -> Dict[str, float]:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``num_params``.
        """
        return dict(
            zip(self.model.params, jnp.dot(self.post, self.grid_param.value).tolist())
        )

    @property
    def post_cov(self) -> jnp.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, num_params)``.
        """
        # shape: (N_grids, N_param)
        d = self.grid_param.value - jnp.array(list(self.post_mean.values()))
        return jnp.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self) -> Dict[str, float]:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``num_params``.
        """
        return dict(zip(self.model.params, jnp.sqrt(jnp.diag(self.post_cov)).tolist()))

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

    def get_design(self, kind="optimal") -> Dict[str, float] | None:
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
        if len(self.grid_design) == 0:
            return None

        if kind == "optimal":
            idx_design = jnp.argmax(self.mutual_info).tolist()

        elif kind == "random":
            idx_design = np.random.randint(self.n_d)

        else:
            raise ValueError('The argument kind should be "optimal" or "random".')

        return self.grid_design[idx_design]

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

        Parameters
        ----------
        design : dict or :code:`pandas.Series` or list of designs
            Design vector for given response
        response : dict or :code:`pandas.Series` or list of responses
            Any kinds of observed response
        """
        assert len(self.task.designs) == len(set(design.keys()))
        assert len(set(self.task.designs) - set(design.keys())) == 0
        assert len(self.task.responses) == len(set(response.keys()))
        assert len(set(self.task.responses) - set(response.keys())) == 0

        d = jnp.array(list(design.values()), dtype=self.dtype)[
            np.argsort(list(design.keys()))
        ]
        y = jnp.array(list(response.values()), dtype=self.dtype)[
            np.argsort(list(response.keys()))
        ]

        i_d, _ = self.grid_design.get_nearest_point(d)
        i_y, _ = self.grid_response.get_nearest_point(y)
        self.log_post = update_log_post(i_d, i_y, self.log_post, self.log_lik)

        self._marg_log_lik = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self._update_mutual_info()
