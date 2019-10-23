from functools import reduce
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np

from adopy.types import data_like
from adopy.functions import extract_vars_from_data

from ._task import Task

__all__ = ['Model']


class Model(object):
    r"""
    A base class for a model in the ADOpy package.

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
    .. todo::
        Update examples using a likelihood function.

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
    >>> model.task
    >>> model.params
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
        model parameters. If the likelihood function is not given for the
        model, it returns a random log probability.

        .. deprecated:: 0.4.0
            As the Model class get modified to contain a function for
            log likelihood in 0.4.0, we recommend you to use
            `model.compute_log_lik()` function instead so to make it clear what
            the function calculates.
        """
        return self.compute_log_lik(*args, **kargs)

    def compute_log_lik(self, *args, **kargs):
        """
        Compute log likelihood of obtaining responses with given designs and
        model parameters. If the likelihood function is not given for the
        model, it returns a random log probability.
        """
        if self._func is not None:
            return self._func(*args, **kargs)

        # If no function is provided, generate a random log probability.
        return np.log(np.random.rand())

    def __repr__(self) -> str:
        strs = []
        strs += 'Model('
        if self.name:
            strs += '{}, '.format(repr(self.name))
        strs += 'params={})'.format(repr(self.params))
        return ''.join(strs)
