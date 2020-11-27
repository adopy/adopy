"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os

import numpy as np

from adopy import base, functions, tasks
from adopy.base import Engine, Model, Task

__version__ = '0.4.0.rc1'

__all__ = [
    # Submodules
    'base', 'functions', 'tasks',
    # Base classes
    'Task', 'Model', 'Engine',
]

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
