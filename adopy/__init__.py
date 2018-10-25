"""
ADOR
====

ADOR is the package of Adaptive Design Optimization wRapper.
The package is based on the code by Woojae Kim and Bryan Zake.
"""
from __future__ import absolute_import, division, print_function

from . import functions
from . import generic
from . import tasks

__all__ = [
    'generic',
    'functions',
    'tasks'
]
