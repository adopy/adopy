"""
ADOpy: Adaptive Design Optimization on Experimental Tasks
"""
import os
import numpy as np

from adopy import base
from adopy import functions
from adopy import tasks

from adopy.base import Task, Model, Engine

__all__ = ['base', 'functions', 'tasks', 'Task', 'Model', 'Engine']

with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as f:
    __version__ = f.read().strip()

# Ignore overflow and underflow floating-point errors
np.seterr(over='ignore', under='ignore')
