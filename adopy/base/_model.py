from typing import Any, Callable, Iterable, Optional, Tuple
import abc

from jax import numpy as jnp

from ._grid import GridSpace
from ._task import Task

__all__ = ["Model"]


class Model(object, metaclass=abc.ABCMeta):
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
    name : Optional[str]
        Name of the task (optional).
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
    grid_param : GridSpace
        Grid space of the model's parameters.
    dtype
        Data type for the grid space. Default to :code:`jax.numpy.float32`.
    """
    __name__ = "Model"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        task: Task,
        params: Iterable[str],
        grid_param: GridSpace,
        dtype: Optional[Any] = jnp.float32,
    ):
        super().__init__()

        self._name = name  # type: Optional[str]
        self._task = task  # type: Task
        self._params = tuple(params)  # type: Tuple[str, ...]
        self._grid_param = grid_param.astype(dtype)
        self._dtype = dtype

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
    def params(self) -> Tuple[str]:
        """Labels for model parameters of the model."""
        return self._params

    @property
    def grid_param(self) -> GridSpace:
        """
        Grid space for design variables, generated from the grid definition,
        given as :code:`grid_design` with initialization.

        .. versionadded:: 0.5.0
        """
        return self._grid_param

    @property
    def dtype(self):
        """
        The desired data-type for the internal vectors and matrixes, e.g.,
        :code:`jax.numpy.float32`. Default is :code:`jax.numpy.float32`.

        .. versionadded:: 0.5.0
        """
        return self._dtype

    @staticmethod
    @abc.abstractmethod
    def compute(*args, **kwargs):
        """
        Compute log likelihood of obtaining responses with given designs and
        model parameters. This function itself is an abstract method, and users
        should write a child class inheriting :py:mod:`adopy.base.Model`,
        with actual implementation of log likelihood function as the compute
        function.

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
        pass

    def __repr__(self) -> str:
        segs = [
            self.__name__,
            "(",
            "name={}, ".format(repr(self.name)) if self.name else "",
            "params={}".format(repr(self.params)),
            ")",
        ]
        return "".join(segs)
