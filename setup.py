from __future__ import unicode_literals

import os
from codecs import open as codecs_open
from setuptools import setup

# Get the long description from the relevant file
with codecs_open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

# Load the version
with open(os.path.join('adopy', 'VERSION'), 'r') as f:
    version = f.read()

setup(
    name='adopy',
    version=version,
    description='',
    long_description=long_description,
    classifiers=[],
    keywords='',
    author='Jaeyeong Yang',
    author_email='jaeyeong.yang1125@gmail.com',
    url='https://github.com/JaeyeongYang/adopy',
    license='GPL-3',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'typing',
        'numpy',
        'pandas',
        'scipy',
    ],
    extras_require={
        'dev': [
            'flake8',
            'pylint',
            'mypy',
            'yapf',
        ],
        'test': [
            'flake8',
            'pylint',
            'pytest',
            'pytest-cov',
            'codecov',
        ],
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-autobuild',
            'travis-sphinx',
        ]
    })
