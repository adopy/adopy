from functools import reduce
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np

from adopy.types import data_like
from adopy.functions import extract_vars_from_data

from ._task import Task

__all__ = ['Model']


class Model(object):
    """
    A base class for a model in the ADOpy package.

    Parameters
    ----------
    task : Task
        Task object that this model object is for.
    params : Iterable[str]
        Labels of model parameters in the model.
    func : Optional[Callable]
        A function to calculate the probability of the observation.
        Currently, it does nothing.
    name : Optional[str]
        Name of the task.

    Examples
    --------
    >>> task = Task(name='Task A', designs=['d1', 'd2'], responses=[0, 1])
    >>> model = Model(name='Model X', task=task, params=['m1', 'm2', 'm3'])
    >>> model
    Model('Model X', params=['m1', 'm2', 'm3'])
    >>> model.name
    'Model X'
    >>> model.task
    Task('Task A', designs=['d1', 'd2'], responses=[0, 1])
    >>> model.params
    ['m1', 'm2', 'm3']
    """

    def __init__(self,
                 task: Task,
                 params: Iterable[str],
                 func: Optional[Callable] = None,
                 constraint: Optional[Dict[str, Callable]] = None,
                 name: Optional[str] = None,
                 ):
        self._name = name  # type: Optional[str]
        self._task = task  # type: Task
        self._params = tuple(params)  # type: Tuple[str, ...]

        self._func = func  # type: Optional[Callable]

        self._constraint = {}  # type: Dict[str, Callable]
        if constraint is not None:
            self._constraint.update(constraint)

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

    @property
    def constraint(self) -> Dict[str, Callable]:
        """Contraints on model parameters. This do not work yet."""
        return self._constraint

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
        """Compute the probability of choosing a certain response given
        values of design variables and model parameters.
        """
        if self._func is not None:
            return self._func(*args, **kargs)
        obj = reduce(lambda x, y: x * y, kargs.values())
        return np.ones_like(obj) / 2

    def __repr__(self) -> str:
        strs = []
        strs += 'Model('
        if self.name:
            strs += '{}, '.format(repr(self.name))
        strs += 'params={})'.format(repr(self.params))
        return ''.join(strs)
