"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import numpy as np

from adopy import base, functions, tasks
from adopy.base import Engine, Model, Task

__all__ = [
    # Submodules
    'base', 'functions', 'tasks',
    # Base classes
    'Task', 'Model', 'Engine',
]

# Load version from the metadata
__version__ = importlib_metadata.version(__name__)

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
