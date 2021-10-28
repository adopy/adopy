from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from jax import numpy as jnp

from adopy.functions import extract_vars_from_data, make_grid_matrix
from adopy.types import data_like

__all__ = ['Task', 'TaskV2']


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
    name
        Name of the task (optional).
    designs
        Labels of design variables in the task.
    responses
        Labels of response variables in the task (e.g., choice, rt).

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
                 *,
                 name: Optional[str] = None,
                 designs: Iterable[str],
                 responses: Iterable[str],
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


class TaskV2(object):
    """
    A task object stores information for a specific experimental task,
    including labels of design variables (:code:`designs`), labels of possible
    responses (:code:`responses`), and the task name (:code:`name`).

    .. versionchanged:: 0.4.0

        The :code:`response` argument is changed to the labels of response
        variables, instead of possible values of a response variable.

    Parameters
    ----------
    name
        Name of the task.
    designs
        Labels of design variables in the task.
    responses
        Labels of response variables in the task (e.g., choice, rt).

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
                 *,
                 name: Optional[str] = None,
                 designs: Iterable[str],
                 responses: Iterable[str],
                 grid_design: Dict[str, Any],
                 grid_response: Dict[str, Any],
                 dtype: Optional[Any] = jnp.float32,
                 ):
        self._name = name
        self._designs = tuple(designs)  # type: Tuple[str, ...]
        self._responses = tuple(responses)  # type: Tuple[str, ...]
        self._dtype = dtype

        self._g_d = jnp.array(
            make_grid_matrix(grid_design)[self.designs].values,
            dtype=self.dtype)
        self._g_y = jnp.array(
            make_grid_matrix(grid_response)[self.responses].values,
            dtype=self.dtype)

    @property
    def name(self) -> Optional[str]:
        """
        Name of the task. If it has no name, returns ``None``.
        """
        return self._name

    @property
    def designs(self) -> Tuple[str]:
        """Labels for design variables of the task."""
        return self._designs

    @property
    def responses(self) -> Tuple[str]:
        """Labels of response variables in the task."""
        return self._responses

    @property
    def grid_design(self) -> pd.DataFrame:
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.
        """
        return pd.DataFrame(self._g_d, columns=self.designs)

    @property
    def grid_response(self):
        """
        Grid space for response variables, generated from the grid definition,
        given as :code:`grid_response` with initialization.
        """
        return pd.DataFrame(self._g_y, columns=self.responses)

    @property
    def dtype(self):
        """
        The desired data-type for the internal vectors and matrixes, e.g.,
        :code:`jax.numpy.float32`. Default is :code:`jax.numpy.float32`.

        TODO: revise the docstring

        .. versionadded:: 0.5.0
        """
        return self._dtype

    def get_grid_design_jax(self):
        """
        TODO: write the docstring
        """
        return self._g_d

    def get_grid_response_jax(self):
        """
        TODO: write the docstring
        """
        return self._g_y

    def __repr__(self) -> str:
        return ''.join([
            'Task(',
            '{}, '.format(repr(self.name)) if self.name else '',
            'designs={}, '.format(repr(self.designs)),
            'responses={})'.format(repr(self.responses))
        ])

    def __eq__(self, other) -> bool:
        return isinstance(other, Task) and \
               self.name == other.name and \
               self.designs == other.designs and \
               self.responses == other.responses
