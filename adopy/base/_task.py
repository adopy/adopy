from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from adopy.functions import extract_vars_from_data
from adopy.types import data_like, number_like, vector_like

__all__ = ['Task']


class Task(object):
    """
    A task object stores information for a specific experimental task,
    including labels of design variables (:code:`designs`), labels of possible
    responses (:code:`responses`), and the task name (:code:`name`).

    .. versionchanged:: 0.4.0

        The :code:`response` argument is changed to the labels of response
        variables, instead of possible values of a response variable.

    Parameters
    ----------
    designs
        Labels of design variables in the task.
    responses
        Labels of response variables in the task (e.g., choice, rt).
    name
        Name of the task.

    Examples
    --------
    >>> task = Task(name='Task A',
    ...             designs=['d1', 'd2'],
    ...             responses=['choice'])
    >>> task
    Task('Task A', designs=['d1', 'd2'], responses=['choice'])
    >>> task.name
    'Task A'
    >>> task.designs
    ['d1', 'd2']
    >>> task.responses
    ['choice']
    """

    def __init__(self,
                 designs: Iterable[str],
                 responses: Iterable[str],
                 name: Optional[str] = None,
                 ):
        self._name = name
        self._designs = tuple(designs)  # type: Tuple[str, ...]
        self._responses = tuple(responses)  # type: Tuple[str, ...]

    @property
    def name(self) -> Optional[str]:
        """
        Name of the task. If it has no name, returns ``None``.
        """
        return self._name

    @property
    def designs(self) -> List[str]:
        """Labels for design variables of the task."""
        return list(self._designs)

    @property
    def responses(self) -> List[str]:
        """Labels of response variables in the task."""
        return list(self._responses)

    def extract_designs(self, data: data_like) -> Dict[str, Any]:
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

    def extract_responses(self, data: data_like) -> Dict[str, Any]:
        """
        Extract response grids from the given data.

        Parameters
        ----------
        data
            A data object that contains key-value pairs or columns
            corresponding to design variables.

        Returns
        -------
        ret
            An ordered dictionary of grids for response variables.
        """
        return extract_vars_from_data(data, self.responses)

    def __repr__(self) -> str:
        strs = []
        strs += 'Task('
        if self.name:
            strs += '{}, '.format(repr(self.name))
        strs += 'designs={}, '.format(repr(self.designs))
        strs += 'responses={})'.format(repr(self.responses))
        return ''.join(strs)

    def __eq__(self, other) -> bool:
        return isinstance(other, Task) and \
            self.name == other.name and \
            self.designs == other.designs and \
            self.responses == other.responses
