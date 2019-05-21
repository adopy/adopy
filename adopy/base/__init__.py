"""
Base classes of ADOpy. These classes provide built-in methods for inherited
classes.
"""
from ._task import Task as _Task
from ._model import Model as _Model
from ._engine import Engine as _Engine

Task = _Task
Model = _Model
Engine = _Engine

__all__ = ['Task', 'Model', 'Engine']
