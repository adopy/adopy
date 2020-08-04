"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os
import numpy as np

from adopy import base, functions, tasks
from adopy import internals, stats

from adopy.base import Task, Model, Engine

__version__ = '0.4.0.rc1'

__all__ = [
    # Submodules
    'base', 'functions', 'tasks',
    # C modules
    'internals', 'stats',
    # Base classes
    'Task', 'Model', 'Engine',
]

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
