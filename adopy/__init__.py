"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os
import numpy as np

from adopy import base
from adopy import functions
from adopy import tasks

from adopy.base import Task, Model, Engine
from .version import version as __version__

__all__ = [
    '__version__',
    # Submodules
    'base', 'functions', 'tasks',
    # Base classes
    'Task', 'Model', 'Engine',
]

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
