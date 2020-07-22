"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os
import numpy as np

from adopy import base
from adopy import functions
from adopy import tasks

from adopy.base import Task, Model, Engine
from adopy import cmodules

__version__ = '0.4.0.rc2'

__all__ = [
    # Submodules
    'base', 'functions', 'tasks',
    # C modules
    'cmodules',
    # Base classes
    'Task', 'Model', 'Engine',
]

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
