"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import numpy as np

from adopy import base, tasks
from adopy.base import GridSpace, Engine, Model, Task

__all__ = [
    # Submodules
    "base",
    "tasks",
    # Base classes
    "GridSpace",
    "Task",
    "Model",
    "Engine",
]

# Load version from the metadata
try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:  # For frozen app support
    __version__ = "0.5.0-dev"

# Ignore overflow and underflow floating-point errors
np.seterr(over="ignore", under="ignore")
