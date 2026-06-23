"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""

import importlib.metadata as importlib_metadata

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
try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:  # For frozen app support
    __version__ = '0.5.0'

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
