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

setup(
    name='adopy',
    url='https://github.com/JaeyeongYang/adopy',
    version=VERSION,
    description='',
    long_description=LONG_DESCRIPTION,
    keywords='',

    author='Jaeyeong Yang',
    author_email='jaeyeong.yang1125@gmail.com',

    license='GPL-3',
    include_package_data=True,
    zip_safe=False,

    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  # noqa: E501
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
