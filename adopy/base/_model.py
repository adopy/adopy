# -*- coding: utf-8 -*-
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from adopy.functions import extract_vars_from_data
from adopy.types import data_like

from ._task import Task

__all__ = ['Model']


class Model(object):
    r"""
    A base class for a model in the ADOpy package.

    Its initialization requires up to 4 arguments: :code:`task`,
    :code:`params`, :code:`func` (optional), and :code:`name` (optional).

    :code:`task` is an instance of a :py:mod:`adopy.base.Task` class that this
    model is for. :code:`params` is a list of model parameters, given as a
    list of their labels, e.g., :code:`['alpha', 'beta']`. :code:`name` is the
    name of this model, which is optional for its functioning.

    The most important argument is :code:`func`, which calculates the log
    likelihood given with design values, parameter values, and response values
    as its input. The arguments of the function should include design variables
    and response variables (defined in the :code:`task`: instance) and model
    parameters (given as :code:`params`). The order of arguments does not
    matter. If :code:`func` is not given, the model provides the log likelihood
    of a random noise. An simple example is given as follows:

    .. code-block:: python

        def compute_log_lik(design1, design2,
                            param1, param2, param3,
                            response1, response2):
            # ... calculating the log likelihood ...
            return log_lik

    .. warning::

        Since the version 0.4.0, the :code:`func` argument should be defined to
        compute the log likelihood, instead of the probability of a binary
        response variable. Also, it should include the response variables as
        arguments. These changes might break existing codes using the previous
        versions of ADOpy.

    .. versionchanged:: 0.4.0

        The :code:`func` argument is changed to the log likelihood function,
        instead of the probability function of a single binary response.

    Parameters
    ----------
    task : Task
        Task object that this model object is for.
    params :
        Labels of model parameters in the model.
    func : function, optional
        A function to compute the log likelihood given a model, denoted as
        :math:`L(\mathbf{x} | \mathbf{d}, \mathbf{\theta})`,
        where :math:`\mathbf{x}` is a response vector,
        :math:`\mathbf{d}` is a design vector, and
        :math:`\mathbf{\theta}` is a parameter vector.
        Note that the function arguments should include all design, parameter,
        and response variables.
    name : Optional[str]
        Name of the task.

    Examples
    --------

    >>> task = Task(name='Task A', designs=['x1', 'x2'], responses=['y'])
    >>> def calculate_log_lik(y, x1, x2, b0, b1, b2):
    ...     import numpy as np
    ...     from scipy.stats import bernoulli
    ...     logit = b0 + b1 * x1 + b2 * x2
    ...     p = np.divide(1, 1 + np.exp(-logit))
    ...     return bernoulli.logpmf(y, p)
    >>> model = Model(name='Model X', task=task, params=['b0', 'b1', 'b2'],
    ...               func=calculate_log_lik)
    >>> model.name
    'Model X'
    >>> model.task
    Task('Task A', designs=['x1', 'x2'], responses=['y'])
    >>> model.params
    ['b0', 'b1', 'b2']
    >>> model.compute(y=1, x1=1, x2=-1, b0=1, b1=0.5, b2=0.25)
    -0.251929081345373
    >>> compute_log_lik(y=1, x1=1, x2=-1, b0=1, b1=0.5, b2=0.25)
    -0.251929081345373
    """

    def __init__(self,
                 task: Task,
                 params: Iterable[str],
                 func: Optional[Callable] = None,
                 name: Optional[str] = None,
                 ):
        self._name = name  # type: Optional[str]
        self._task = task  # type: Task
        self._params = tuple(params)  # type: Tuple[str, ...]

        if func is not None:
            func_args = func.__code__.co_varnames

            # Check if the function arguments are valid.
            if not (all([x in func_args for x in task.responses])
                    and all([d in func_args for d in task.designs])
                    and all([p in func_args for p in params])):
                raise RuntimeError(
                    'Argument names of the likelihood function should contain '
                    'those of design, parameter, response variables.')

        self._func = func  # type: Optional[Callable]

    @property
    def name(self) -> Optional[str]:
        """
        Name of the model. If it has no name, returns ``None``.
        """
        return self._name

    @property
    def task(self) -> Task:
        """Task instance for the model."""
        return self._task

    @property
    def params(self) -> List[str]:
        """Labels for model parameters of the model."""
        return list(self._params)

    def extract_params(self, data: data_like) -> Dict[str, Any]:
        """
        Extract parameter grids from the given data.

        Parameters
        ----------
        data
            A data object that contains key-value pairs or columns
            corresponding to design variables.

        Returns
        -------
        ret
            An ordered dictionary of grids for model parameters.
        """
        return extract_vars_from_data(data, self.params)

    def compute(self, *args, **kargs):
        """
        Compute log likelihood of obtaining responses with given designs and
        model parameters. The function provide the same result as the argument
        :code:`func` given in the initialization. If the likelihood function is
        not given for the model, it returns the log probability of a random
        noise.

        .. warning::

            Since the version 0.4.0, :code:`compute()` function should compute
            the log likelihood, instead of the probability of a binary response
            variable. Also, it should include the response variables as
            arguments. These changes might break existing codes using the
            previous versions of ADOpy.

        .. versionchanged:: 0.4.0

            Provide the log likelihood instead of the probability of a binary
            response.
        """
        if self._func is not None:
            return self._func(*args, **kargs)

        # If no function is provided, generate a random log probability.
        return np.log(np.random.rand())

    def __repr__(self) -> str:
        strs: List[str] = []
        strs += 'Model('
        if self.name:
            strs += '{}, '.format(repr(self.name))
        strs += 'params={})'.format(repr(self.params))
        return ''.join(strs)
