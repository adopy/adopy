from __future__ import absolute_import, division, print_function

import abc
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable

__all__ = ['Task', 'Model']


class InvalidArgumentError(BaseException):
    def __init__(self, message):
        super(InvalidArgumentError, self).__init__(message)


class MetaInterface(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, var):
        # type: (str, Iterable[str]) -> None
        self._name = name
        self._var = tuple(var)

    @property
    def name(self):
        return self._name

    @property
    def var(self):
        return self._var

    def get_vars_from_dict(self, d):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        ret = {}  # type: Dict[str, Any]
        for v in self.var:
            ret[v] = d.get(v, None)
        return ret


class Task(MetaInterface):
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return 'Task(name={name}, var={var})'\
            .format(name=repr(self.name), var=repr(list(self.var)))


class Model(MetaInterface):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, var, task, func):
        # type: (str, Iterable[str], Task, Callable) -> None
        super(Model, self).__init__(name, var)
        self._task = task
        self._func = func

    @property
    def task(self):
        return self._task

    @property
    def func(self):
        return self._func

    def compute(self, **kargs):
        args = {}
        args.update(self.get_vars_from_dict(kargs))
        args.update(self.task.get_vars_from_dict(kargs))
        return self.func(**args)

    def __repr__(self):
        return 'Model(name={name}, var={var}, task={task})'\
            .format(name=repr(self.name), var=repr(list(self.var)), task=repr(self.task))
