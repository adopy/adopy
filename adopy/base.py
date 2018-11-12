"""
Base Classes
============

"""
from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar
from collections import OrderedDict
from functools import reduce

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal as mvnm
import pandas as pd

from adopy.functions import (expand_multiple_dims, get_nearest_grid_index,
                             get_random_design_index, make_grid_matrix,
                             marginalize, make_vector_shape, log_lik_bernoulli)

__all__ = ['Task', 'Model', 'Engine']

DT = TypeVar('DT', Dict[str, Any], pd.DataFrame)


class MetaInterface(object):
    """
    Meta interface for tasks and models.

    Generate a singleton instance.
    """
    _instance = None  # type: object

    def __init__(self, name, key):
        # type: (str, str) -> None
        self._name = name  # type: str
        self._key = key  # type: str

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    @property
    def key(self):  # type: () -> str
        """Key for the meta instance"""
        return self._key

    @property
    def name(self):  # type: () -> str
        """Name for the meta instance"""
        return self._name

    @staticmethod
    def _extract_vars(dt, keys):
        # type: (DT, Iterable[str]) -> OrderedDict[str, Any]
        ret = OrderedDict()  # type: OrderedDict[str, Any]

        for k in keys:
            ret[k] = dt[k] if isinstance(dt, dict) else dt[k].values

        return ret


class Task(MetaInterface):
    """
    Metaclass for tasks

    >>> task = Task('Task A', 'a', ['d1', 'd2'])
    >>> task
    Task('Task A', design=['d1', 'd2'])
    """

    def __init__(self, name, key, design):
        # type: (str, str, Iterable[str]) -> None
        super(Task, self).__init__(name, key)
        self._design = tuple(design)  # type: Tuple[str, ...]

    @property
    def design(self):  # type: () -> List[str]
        """Design labels of the task"""
        return list(self._design)

    def extract_designs(self, dt):
        # type: (DT) -> OrderedDict[str, Any]
        """
        Extract design grids from given dictionary or dataframe.

        Parameters
        ----------
        dt : Dict[str, array_like] or pd.DataFrame

        Returns
        -------
        OrderedDict[str, array_like]
            An ordered dictionary of single grids for design variables.
        """
        return self._extract_vars(dt, self.design)

    def __repr__(self):  # type: () -> str
        return 'Task({name}, design={var})' \
            .format(name=repr(self.name), var=repr(list(self.design)))


class Model(MetaInterface):
    """
    Metaclass for models

    >>> task = Task('Task A', 'a', ['d1', 'd2'])
    >>> model = Model('Model X', 'x', task, ['m1', 'm2', 'm3'])
    >>> model
    Model('Model X', param=['m1', 'm2', 'm3'])
    """

    def __init__(self,
                 name,            # type: str
                 key,             # type: str
                 task,            # type: Task
                 param,           # type: Iterable[str]
                 func=None,       # type: Optional[Callable]
                 constraint=None  # type: Optional[Dict[str, Callable]]
                 ):
        # type: (...) -> None
        super(Model, self).__init__(name, key)

        self._task = task  # type: Task
        self._param = tuple(param)  # type: Tuple[str, ...]

        def _func(**kargs):
            if func is not None:
                return func(**kargs)
            obj = reduce(lambda x, y: x * y, kargs.values())
            return np.ones_like(obj) / 2

        self._func = _func  # type: Callable

        self._constraint = {}  # type: Dict[str, Callable]
        if constraint is not None:
            self._constraint.update(constraint)

    @property
    def task(self):  # type: () -> Task
        """Task instance for the model"""
        return self._task

    @property
    def param(self):  # type: () -> List[str]
        """Parameter labels of the model"""
        return list(self._param)

    @property
    def constraint(self):  # type: () -> Dict[str, Callable]
        """Contraints on model parameters"""
        return self._constraint

    def extract_params(self, dt):
        # type: (DT) -> OrderedDict[str, Any]
        return self._extract_vars(dt, self.param)

    def compute(self, **kargs):
        # type: (...) -> Any
        return self._func(**kargs)

    def __repr__(self):  # type: () -> str
        return 'Model({name}, param={var})' \
            .format(name=repr(self.name), var=repr(list(self.param)))


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

    def __init__(self, task, model, designs, params, y_obs):
        super(Engine, self).__init__()

        if model.task is not task:
            raise AssertionError('Given task and model are not matched.')

        self._task = task  # type: Task
        self._model = model  # type: Model

        self.grid_design = make_grid_matrix(designs)[list(task.design)]
        self.grid_param = make_grid_matrix(params)[list(model.param)]
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

        self.flag_update_mutual_info = True

    ###########################################################################
    # Properties
    ###########################################################################

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
        return [
            marginalize(self.post, self.grid_param, i)
            for i in range(self.num_param)
        ]

    @property
    def post_mean(self):  # type: () -> np.ndarray
        """Estimated posterior means for each parameter"""
        return np.dot(self.post, self.grid_param)

    @property
    def post_cov(self):  # type: () -> np.ndarray
        # shape: (N_grids, N_param)
        d = self.grid_param.values - self.post_mean
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self):
        return np.sqrt(np.diag(self.post_cov))

    ###########################################################################
    # Methods
    ###########################################################################

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

        1. :code:`optimal`: an optimal design :math:`d^*` that maximizes the
            mutual information.

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
        if kind not in {'optimal', 'random'}:
            raise AssertionError(
                'The argument kind should be "optimal" or "random".')

        if kind == 'optimal':
            self._update_mutual_info()
            return self.grid_design.iloc[np.argmax(self.mutual_info)]

        if kind == 'random':
            return self.grid_design.iloc[get_random_design_index(
                self.grid_design)]

        raise AssertionError('An invalid kind of design: "{}".'.format(type))

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
        idx_response = get_nearest_grid_index(
            pd.Series(response), self.grid_response)

        self.log_post += self.log_lik[idx_design, :, idx_response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self.flag_update_mutual_info = True

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
        """
        Update the grid space for model parameters (Dynamic Gridding method)
        """
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
            idx = self.model.param.index(k)
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
