"""
ADOpy
=====

ADOpy is a Python package for Adaptive Design Optimization.
"""
from adopy import base
from adopy import functions
from adopy import tasks

from adopy.base import Task, Model, Engine
from adopy.version import VERSION

__all__ = ['base', 'functions', 'tasks', 'Task', 'Model', 'Engine']
__version__ = VERSION
