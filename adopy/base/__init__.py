"""
This module contains three basic classes of ADOpy: `Task`, `Model`, and `Engine`.
These classes provide built-in functions for the Adaptive Design Optimization.

.. note::

   Three basic classes are defined in the :py:mod:`adopy.base` (i.e.,
   :py:class:`adopy.base.Task`, :py:class:`adopy.base.Model`, and
   :py:class:`adopy.base.Engine`). However, for convinience, users can find it
   directly as :py:class:`adopy.Task`, :py:class:`adopy.Model`, and
   :py:class:`adopy.Engine`.
"""
from ._task import Task
from ._model import Model
from ._engine import Engine

__all__ = ['Task', 'Model', 'Engine']
