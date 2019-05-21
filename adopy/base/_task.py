from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np

from adopy.types import TYPE_NUMBER, TYPE_DATA, TYPE_VECTOR
from adopy.functions import extract_vars_from_data

from ._meta import MetaInterface

__all__ = ['Task']


class Task(MetaInterface):
    """
    A base class for a task in the ADOpy package.

    Parameters
    ----------
    name : str
        Name of the task.
    designs : Iterable[str]
        Labels for design variables of the task.
    responses : Iterable[TYPE_NUMBER]
        Possible values for the response variable of the task.
    key : Optional[str]
        A hash key for the task.

    Examples
    --------
    >>> task = Task(name='Task A', designs=['d1', 'd2'], responses=[0, 1])
    >>> task
    Task('Task A', designs=['d1', 'd2'], responses=[0, 1])
    >>> task.name
    'Task A'
    >>> task.designs
    ['d1', 'd2']
    >>> task.responses
    [0, 1]
    """

    def __init__(self,
                 name: str,
                 designs: Iterable[str],
                 responses: Iterable[TYPE_NUMBER],
                 key: str = None,
                 ):
        super(Task, self).__init__(name, key)
        self._designs = tuple(designs)  # type: Tuple[str, ...]
        self._responses = np.array(responses)  # type: TYPE_VECTOR

    @property
    def designs(self) -> List[str]:
        """Labels for design variables of the task."""
        return list(self._designs)

    @property
    def responses(self) -> List[str]:
        """Possible values for the response variable of the task."""
        return list(self._responses)

    def extract_designs(self, data: TYPE_DATA) -> Dict[str, Any]:
        """
        Extract design grids from the given data.

        Parameters
        ----------
        data
            A data object that contains key-value pairs or columns
            corresponding to design variables.

        Returns
        -------
        ret
            An ordered dictionary of grids for design variables.
        """
        return extract_vars_from_data(data, self.designs)

    def __repr__(self) -> str:
        return 'Task({name}, designs={designs}, responses={responses})'.format(
            name=repr(self.name),
            designs=repr(self.designs),
            responses=repr(self.responses))
