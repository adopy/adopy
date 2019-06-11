from functools import reduce
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np

from adopy.types import data_like
from adopy.functions import extract_vars_from_data

from ._meta import MetaInterface
from ._task import Task

__all__ = ['Model']


class Model(MetaInterface):
    """
    A base class for a model in the ADOpy package.

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
        super(Model, self).__init__(name)

        self._task = task  # type: Task
        self._params = tuple(params)  # type: Tuple[str, ...]

        self._func = func  # type: Optional[Callable]

        self._constraint = {}  # type: Dict[str, Callable]
        if constraint is not None:
            self._constraint.update(constraint)

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
        """Contraints on model parameters"""
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

    def compute(self, **kargs):
        """
        Compute the likelihood of data given model parameters.
        """
        if self._func is not None:
            return self._func(**kargs)
        obj = reduce(lambda x, y: x * y, kargs.values())
        return np.ones_like(obj) / 2

    def __repr__(self) -> str:
        return 'Model({name}, params={params})'.format(
            name=repr(self.name),
            params=repr(self.params))
