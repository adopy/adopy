from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal as mvnm
import pandas as pd

from adopy.functions import (expand_multiple_dims, get_nearest_grid_index, get_random_design_index, make_grid_matrix,
                             marginalize, make_vector_shape, log_lik_bernoulli)

__all__ = ['Engine']


class Engine(object):
    """Generic class for ADOpy classes.

    Examples
    --------

    .. code-block:: python3
        :linenos:

        ado =
        task = Task('Task A', 'a', ['d1', 'd2'])
        model = Model('Model X', 'x', ['m1', 'm2', 'm3'])
        for _ in range(num_trials):  # Loop for trials
            design = ado.get_design()
            response = get_response(design)
            ado.update(design, response)

    get_response functions is a pseudo-function to run an experiment and get
    a response.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, task, model, designs, params, y_obs):
        super(Engine, self).__init__()

        if model.task is not task:
            raise RuntimeError('Given task and model are not matched.')
        self._task = task  # type: Task
        self._model = model  # type: Model

        self.grid_design = make_grid_matrix(designs)[list(task.design)]
        self.grid_param = make_grid_matrix(params)[list(model.param)]
        # TODO: consider cases with multiple response variables
        self.grid_response = pd.DataFrame(np.array(y_obs), columns=['y_obs'])

        self.y_obs = np.array(y_obs)
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

        # For dynamic gridding
        self.dg_memory = []  # [(design, response), ...]
        self.dg_grid_params = []
        self.dg_means = []
        self.dg_covs = []
        self.dg_priors = []
        self.dg_posts = []

        self.idx_opt = None
        self.design_opt = None

        self.flag_update_mutual_info = True

    ##################################################################################################################
    # Properties
    ##################################################################################################################

    task = property(lambda self: self._task)
    """Task: Task of the engine"""

    model = property(lambda self: self._model)
    """Model: Model of the engine"""

    num_design = property(lambda self: len(self.task.design))
    """Number of design grid axes"""

    num_param = property(lambda self: len(self.model.param))
    """Number of parameter grid axes"""

    prior = property(lambda self: np.exp(self.log_prior))
    """Prior distributions of joint parameter space"""

    post = property(lambda self: np.exp(self.log_post))
    """Posterior distributions of joint parameter space"""

    @property
    def marg_post(self):
        """Marginal posterior distributions for each parameter"""
        return [marginalize(self.post, self.grid_param, i) for i in range(self.num_param)]

    post_mean = property(lambda self: np.dot(self.post, self.grid_param))
    """Estimated posterior means for each parameter"""

    @property
    def post_cov(self):
        m = self.post_mean
        d = self.grid_param - m
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self):
        return np.sqrt(np.diag(self.post_cov))

    ##################################################################################################################
    # Methods
    ##################################################################################################################

    def _initialize(self):
        self.p_obs = self._compute_p_obs()
        self.log_lik = ll = self._compute_log_lik()
        self.ent_obs = -np.multiply(np.exp(ll), ll).sum(-1)

        lp = np.ones(self.grid_param.shape[0])
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

    def _compute_p_obs(self):
        """Compute the probability of getting observed response."""
        shape_design = make_vector_shape(2, 0)
        shape_param = make_vector_shape(2, 1)

        args = {}
        args.update({k: v.reshape(shape_design) for k, v in self.task.extract_designs(self.grid_design).items()})
        args.update({k: v.reshape(shape_param) for k, v in self.model.extract_params(self.grid_param).items()})

        return self.model.compute(**args)

    def _compute_log_lik(self):
        """Compute the log likelihood."""
        # TODO: Cover the case for Categorical distribution
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
        self.ent_cond = np.sum(self.post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - self.ent_cond  # shape (num_designs,)

        # Flag that there is no need to update mutual information again.
        self.flag_update_mutual_info = False

    def get_design(self, kind='optimal'):
        # type: (str) -> np.ndarray
        r"""
        Choose a design with a given type.

        1. :code:`optimal`: an optimal design :math:`d^*` that maximizes the mutual information.

            .. math::
                \begin{align*}
                    p(y | d) &= \sum_\theta p(y | \theta, d) p_t(\theta) \\
                    I(Y(d); \Theta) &= H(Y(d)) - H(Y(d) | \Theta) \\
                    d^* &= \operatorname*{argmax}_d I(Y(d); |Theta) \\
                \end{align*}

        2. :code:`random`: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : array_like
            A chosen design vector
        """
        assert kind in {'optimal', 'random'}

        def get_design_optimal():
            return self.grid_design.iloc[np.argmax(self.mutual_info)]

        def get_design_random():
            return self.grid_design.iloc[get_random_design_index(self.grid_design)]

        if kind == 'optimal':
            self._update_mutual_info()
            return get_design_optimal()
        elif kind == 'random':
            return get_design_random()
        else:
            raise RuntimeError('An invalid kind of design: "{}".'.format(type))

    def update(self, design, response, store=True):
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
        store : bool
            Whether to store observations of (design, response).
        """
        if store:
            self.dg_memory.append((design, response))

        if not isinstance(design, pd.Series):
            design = pd.Series(design, index=self.task.design)

        idx_design = get_nearest_grid_index(design, self.grid_design)
        idx_response = get_nearest_grid_index(pd.Series(response), self.grid_response)

        self.log_post += self.log_lik[idx_design, :, idx_response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self.flag_update_mutual_info = True

    def update_grid(self, grid, rotation='eig', grid_type='q', prior='normal', append=False):
        """
        Update the grid space for model parameters (Dynamic Gridding method)
        """
        assert rotation in {'eig', 'svd', 'none', None}
        assert grid_type in {'q', 'z'}
        assert prior in {'recalc', 'normal', None}

        m = self.post_mean
        cov = self.post_cov
        sd = self.post_sd

        if np.linalg.det(cov) == 0:
            print('Cannot update grid no more.')
            return

        # Calculate a rotation matrix
        r_inv = None
        if rotation == 'eig':
            el, ev = np.linalg.eig(cov)
            r_inv = np.dot(np.sqrt(np.diag(el)), np.linalg.inv(ev))
        elif rotation == 'svd':
            _, sg, sv = np.linalg.svd(cov)
            r_inv = np.dot(np.sqrt(np.diag(sg)), sv)
        elif rotation == 'none' or rotation is None:
            r_inv = np.diag(sd)

        # Find grid points from the rotated space
        g_axes = None
        if grid_type == 'q':
            assert all([0 <= v <= 1 for v in grid])
            g_axes = np.repeat(norm.ppf(np.array(grid)).reshape(-1, 1), self.num_param, axis=1)
        elif grid_type == 'z':
            g_axes = np.repeat(np.array(grid).reshape(-1, 1), self.num_param, axis=1)

        # Compute new grid on the initial space.
        g_star = make_grid_matrix(*[v for v in g_axes.T])
        grid_new = np.dot(g_star, r_inv) + m

        # Remove improper points not in the parameter space
        for k, f in self.model.constraint.items():
            idx = self.model.param.index(k)
            grid_new = grid_new[list(map(f, grid_new[:, idx]))]

        self.dg_means.append(m)
        self.dg_covs.append(cov)
        self.dg_grid_params.append(grid_new)
        if append:
            self.grid_param = np.concatenate([self.grid_param, grid_new])
        else:
            self.grid_param = grid_new

        self.dg_priors.append(self.prior)
        self.dg_posts.append(self.post)
        if append:
            log_post_prev = self.log_post.copy()

        self._initialize()

        # Assign priors on new grid
        if prior == 'recalc':
            for d, y in self.dg_memory:
                self.update(d, y, False)
        elif prior == 'normal':
            if append:
                mvnm_prior = mvnm.pdf(grid_new, mean=m, cov=cov)
                self.log_prior = np.concatenate([log_post_prev, mvnm_prior])
                self.log_post = self.log_prior.copy()
            else:
                self.log_prior = mvnm.pdf(grid_new, mean=m, cov=cov)
                self.log_post = self.log_prior.copy()
        elif prior is None:
            pass
