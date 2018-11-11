import os

__all__ = ['VERSION']

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')

with open(VERSION_FILE, 'r') as f:
    VERSION = f.read()
