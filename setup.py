from __future__ import unicode_literals

import os
from codecs import open as codecs_open
from setuptools import setup

# Get the long description from the relevant file
with codecs_open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Load the version
with open(os.path.join('adopy', 'VERSION'), 'r') as f:
    VERSION = f.read().strip()

# Setup with minimal arguments
setup(
    name='adopy',
    version=VERSION,
    url='https://github.com/adopy',
    description='',
    author='Jaeyeong Yang',
    author_email='jaeyeong.yang1125@gmail.com',
    license='GPL-3',
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
)
