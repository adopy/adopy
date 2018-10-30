from __future__ import absolute_import, division, print_function

import abc
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar
from collections import OrderedDict

import pandas as pd

__all__ = ['Task', 'Model']

DT = TypeVar('DT', Dict[str, Any], pd.DataFrame)


class MetaInterface(object):
    """Meta interface for tasks and models.

    Generate a singleton instance.
    """
    _instance = None  # type: object

    def __init__(self, name, key):
        # type: (str, str) -> None
        self._name = name  # type: str
        self._key = key  # type: str

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    key = property(lambda self: self._key)
    """str: Key for the meta instance"""

    name = property(lambda self: self._name)
    """str: name of the meta instance"""

    def _extract_vars(self, dt, keys):
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
    Task('Task A', var=['d1', 'd2'])
    """

    def __init__(self, name, key, design):
        # type: (str, str, Iterable[str]) -> None
        super(Task, self).__init__(name, key)
        self._design = tuple(design)  # type: Tuple[str, ...]

    design = property(lambda self: self._design)
    """Tuple[str]: Design labels of the task"""

    def extract_designs(self, dt):
        # type: (DT) -> OrderedDict[str, Any]
        return self._extract_vars(dt, self.design)

    def __repr__(self):  # type: () -> str
        return 'Task({name}, design={var})'\
            .format(name=repr(self.name), var=repr(list(self.design)))


class Model(MetaInterface):
    """
    Metaclass for models

    >>> task = Task('Task A', 'a', ['d1', 'd2'])
    >>> model = Model('Model X', 'x', ['m1', 'm2', 'm3'])
    >>> model
    Model('Model X', var=['m1', 'm2', 'm3'])
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, key, task, param, constraint=None):
        # type: (str, str, Task, Iterable[str], Optional[Dict[str, Callable]]) -> None
        super(Model, self).__init__(name, key)
        self._task = task  # type: Task
        self._param = tuple(param)  # type: Tuple[str, ...]
        self._constraint = {} if constraint is None else constraint  # type: Dict[str, Callable]

    task = property(lambda self: self._task)
    """Task: Task instance for the model"""

    param = property(lambda self: self._param)
    """Tuple[str]: Parameter labels of the model"""

    constraint = property(lambda self: self._constraint)
    """Dict[str, Callable]: Contraints on model parameters"""

    def extract_params(self, dt):
        # type: (DT) -> OrderedDict[str, Any]
        return self._extract_vars(dt, self.param)

    @abc.abstractmethod
    def compute(self, **kargs):
        # type: (...) -> Any
        raise NotImplementedError('Model.compute method is not implemented.')

    def __repr__(self):  # type: () -> str
        return 'Model({name}, param={var})'\
            .format(name=repr(self.name), var=repr(list(self.param)))
