from typing import Any, Iterable, Optional, Tuple
import abc

from jax import numpy as jnp

from .grid import GridSpace

__all__ = ["AbstractTask", "Task"]


class AbstractTask(object, metaclass=abc.ABCMeta):
    """
    An abstract class representing a specific experimental task, without grid
    definitions. It includes following 3 information:
    1. :code:`name` (name of the task, optional),
    2. :code:`designs` (a list of labels for design variables), and
    3. :code:`responses` (a list of labels for response variables).

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
    >>> abt = AbstractTask(name='Task A', designs=['d1', 'd2'], responses=['y'])
    >>> abt.name
    'Task A'
    >>> abt.designs
    ('d1', 'd2')
    >>> abt.responses
    ('y',)
    """

    __name__ = "AbstractTask"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        designs: Iterable[str],
        responses: Iterable[str],
    ):
        super().__init__()

        self._name = name
        self._designs = tuple(designs)  # type: Tuple[str, ...]
        self._responses = tuple(responses)  # type: Tuple[str, ...]

    @property
    def name(self) -> Optional[str]:
        """Name of the task. If it has no name, returns ``None``."""
        return self._name

    @property
    def designs(self) -> Tuple[str]:
        """Labels for design variables of the task."""
        return self._designs

    @property
    def responses(self) -> Tuple[str]:
        """Labels of response variables in the task."""
        return self._responses

    def __repr__(self) -> str:
        return "".join(
            [
                self.__name__,
                "(",
                "name={}, ".format(repr(self.name)) if self.name else "",
                "designs={}, ".format(repr(self.designs)),
                "responses={}".format(repr(self.responses)),
                ")",
            ]
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Task)
            and self.name == other.name
            and self.designs == other.designs
            and self.responses == other.responses
        )


class Task(AbstractTask):
    """
    A class representing a specific experimental task including grid
    definitions. It includes following 6 information:
    1. :code:`name` (name of the task, optional),
    2. :code:`designs` (a list of labels for design variables),
    3. :code:`responses` (a list of labels for response variables),
    4. :code:`grid_design` (a grid space for design variables),
    5. :code:`grid_response` (a grid space for response variables), and
    6. :code:`dtype` (data type for grid spaces, optional)

    Parameters
    ----------
    name : Optional[str]
        Name of the task. Default is ``None``.
    designs : List[str]
        Labels of design variables in the task.
    responses : List[str]
        Labels of response variables in the task (e.g., choice, rt).
    grid_design : GridSpace
        Grid space for design variables.
    grid_response : GridSpace
        Grid space for response variables.
    dtype : Optional[Any]
        data type of grid spaces (optional). Default is ``float32``.
    """

    __name__ = "Task"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        designs: Iterable[str],
        responses: Iterable[str],
        grid_design: GridSpace,
        grid_response: GridSpace,
        dtype: Optional[Any] = jnp.float32,
    ):
        super(Task, self).__init__(
            name=name,
            designs=designs,
            responses=responses,
        )

        self._grid_design = grid_design.astype(dtype)
        self._grid_response = grid_response.astype(dtype)
        self._dtype = dtype

    @property
    def grid_design(self) -> GridSpace:
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.
        """
        return self._grid_design

    @property
    def grid_response(self) -> GridSpace:
        """
        Grid space for response variables, generated from the grid definition,
        given as :code:`grid_response` with initialization.
        """
        return self._grid_response

    @property
    def dtype(self):
        """
        The desired data-type for the internal vectors and matrixes, e.g.,
        :code:`jax.numpy.float32`. Default is :code:`jax.numpy.float32`.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._grid_design = self._grid_design.astype(value)
        self._grid_response = self._grid_response.astype(value)
        self._dtype = value

    def to_abstract_task(self):
        """Convert Task into :py:mod:`adopy.base.AbstractTask`."""
        return AbstractTask(
            name=self.name, designs=self.designs, responses=self.responses
        )
