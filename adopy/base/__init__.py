# -*- coding: utf-8 -*-
"""
This module contains three basic classes of ADOpy: `Task`, `Model`, and `Engine`.
These classes provide built-in functions for the Adaptive Design Optimization.

.. note::

   Three basic classes are defined in the :py:mod:`adopy.base` (i.e.,
   :py:class:`adopy.base.Task`, :py:class:`adopy.base.Model`, and
   :py:class:`adopy.base.Engine`). However, for convinience, users can import them
   directly as :py:class:`adopy.Task`, :py:class:`adopy.Model`, and
   :py:class:`adopy.Engine`.

   .. code-block:: python

      from adopy import Task, Model, Engine
      # works the same as
      from adopy.base import Task, Model, Engine
"""
from .grid import GridSpace
from .task import AbstractTask, Task
from .model import Model
from .engine import Engine

__all__ = [
    "GridSpace",
    "AbstractTask",
    "Task",
    "Model",
    "Engine",
]
