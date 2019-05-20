from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal as mvnm

from adopy.functions import (
    make_grid_matrix,
)

from ._task import Task
from ._model import Model
from ._engine import Engine

__all__ = ['EngineDG']


class EngineDG(Engine):
    """
    A base class for an ADO engine to compute optimal designs.
    """

    def __init__(self,
                 task: Task,
                 model: Model,
                 designs: Dict[str, Any],
                 params: Dict[str, Any]):
        super(EngineDG, self).__init__(task, model, designs, params)

        # For dynamic gridding
        self.dg_memory = []  # [(design, response), ...]
        self.dg_grid_params = []
        self.dg_means = []
        self.dg_covs = []
        self.dg_priors = []
        self.dg_posts = []

    ###########################################################################
    # Methods
    ###########################################################################

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

        super(EngineDG, self).update(design, response)

    def _get_rotation_matrix(self, rotation):
        if rotation not in {'eig', 'svd', 'none', None}:
            raise AssertionError(
                'rotation should be "eig", "svd", "none", or None.')

        if rotation == 'eig':
            el, ev = np.linalg.eig(self.post_cov)
            ret = np.dot(np.sqrt(np.diag(el)), np.linalg.inv(ev))
        elif rotation == 'svd':
            _, sg, sv = np.linalg.svd(self.post_cov)
            ret = np.dot(np.sqrt(np.diag(sg)), sv)
        else:
            ret = np.diag(self.post_sd)

        return ret

    def _get_grid_axes(self, grid, grid_type):
        if grid_type not in {'q', 'z'}:
            raise AssertionError(
                'grid_type should be "q" (quantiles) or "z" (Z scores).')

        if grid_type == 'q':
            if not all([0 <= v <= 1 for v in grid]):
                raise AssertionError(
                    'All quantile values should be between 0 and 1.')
            g_axes = np.repeat(
                norm.ppf(np.array(grid)).reshape(-1, 1),
                self.num_param,
                axis=1)
        elif grid_type == 'z':
            g_axes = np.repeat(
                np.array(grid).reshape(-1, 1), self.num_param, axis=1)

        return g_axes

    def update_grid(self,
                    grid,
                    rotation='eig',
                    grid_type='q',
                    prior='normal',
                    append=False):
        # Update the grid space for model parameters (Dynamic Gridding method)
        if rotation not in {'eig', 'svd', 'none', None}:
            raise AssertionError('Invalid argument: rotation')

        if grid_type not in {'q', 'z'}:
            raise AssertionError('Invalid argument: grid_type')

        if prior not in {'recalc', 'normal', None}:
            raise AssertionError('Invalid argument: prior')

        m = self.post_mean
        cov = self.post_cov

        if np.linalg.det(cov) == 0:
            print('Cannot update grid no more.')
            return

        # Calculate a rotation matrix
        r_inv = self._get_rotation_matrix(rotation)

        # Find grid points from the rotated space
        g_axes = self._get_grid_axes(grid, grid_type)

        # Compute new grid on the initial space.
        g_star = make_grid_matrix(*[v for v in g_axes.T])
        grid_new = np.dot(g_star, r_inv) + m

        # Remove improper points not in the parameter space
        for k, f in self.model.constraint.items():
            idx = self.model.params.index(k)
            grid_new = grid_new[list(map(f, grid_new[:, idx]))]

        self.grid_param = np.concatenate([self.grid_param, grid_new])\
            if append else grid_new

        self.dg_means.append(m)
        self.dg_covs.append(cov)
        self.dg_grid_params.append(grid_new)
        self.dg_priors.append(self.prior)
        self.dg_posts.append(self.post)

        self._initialize()

        # Assign priors on new grid
        if prior == 'recalc':
            for d, y in self.dg_memory:
                self.update(d, y, False)

        elif prior == 'normal':
            if append:
                mvnm_prior = mvnm.pdf(grid_new, mean=m, cov=cov)
                self.log_prior = np.concatenate(
                    [self.log_post.copy(), mvnm_prior])
            else:
                self.log_prior = mvnm.pdf(grid_new, mean=m, cov=cov)

            self.log_post = self.log_prior.copy()

        elif prior is None:
            pass
