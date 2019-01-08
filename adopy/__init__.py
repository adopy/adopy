"""
ADOpy
=====

ADOpy is a Python package for Adaptive Design Optimization.
"""
import os

from adopy import base
from adopy import functions
from adopy import tasks

from adopy.base import Task, Model, Engine

__all__ = ['base', 'functions', 'tasks', 'Task', 'Model', 'Engine']

with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as f:
    __version__ = f.read().strip()
