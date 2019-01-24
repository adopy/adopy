"""
Adaptive Design Optimization on Experimental Tasks
"""
import os
from codecs import open as codecs_open
from setuptools import setup, find_packages

# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Load the version
with open(os.path.join('adopy', 'VERSION'), 'r') as f:
    VERSION = f.read().strip()

setup(
    name='adopy',
    version=VERSION,
    url='https://adopy.org/',
    description='Adaptive Design Optimization on Experimental Tasks',
    long_description=LONG_DESCRIPTION,
    author='Jaeyeong Yang',
    author_email='jaeyeong.yang1125@gmail.com',
    license='GPL-3',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
