.. raw:: html

    <p align="center">
      <img src="https://user-images.githubusercontent.com/11037140/44476928-3b9e1d80-a607-11e8-8fe9-b2e4758e92ec.png"
           alt="ADOpy: Adaptive Design Optimization for Psychological Tasks"
           width="300px" height="150px">
    </p>

.. raw:: html

    <p align="center">
      <a href="https://www.repostatus.org/#wip">
        <img src="https://www.repostatus.org/badges/latest/wip.svg"
             alt="Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public." />
      </a>
      <a href="https://travis-ci.com/JaeyeongYang/adopy">
        <img src="https://travis-ci.com/JaeyeongYang/adopy.svg?token=gbyEQoyAYgexeSRwBwj6&branch=master" alt="Travis CI" />
      </a>
      <a href="https://codecov.io/gh/JaeyeongYang/adopy">
        <img src="https://codecov.io/gh/JaeyeongYang/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k" alt="CodeCov" />
      </a>
      <a href="https://www.codefactor.io/repository/github/jaeyeongyang/adopy">
        <img src="https://www.codefactor.io/repository/github/jaeyeongyang/adopy/badge" alt="CodeFactor" />
      </a>
    </p>

ADOpy is a Python package for Adaptive Design Optimization on experimental
tasks.

Dependencies
------------

- Python 3.5+
- NumPy
- Pandas
- SciPy

Installation
------------

.. code-block:: bash

    # Clone the repository from Github.
    git clone https://github.com/JaeyeongYang/adopy.git

    # Set the working directory to the cloned repository.
    cd adopy

    # Install ADOpy with pip
    pip install .

Development
-----------

You can set up a developmental environment using pipenv.

.. code-block:: bash

   # Clone the repository from Github.
   git clone https://github.com/JaeyeongYang/adopy.git

   # Set the working directory to the cloned repository.
   cd adopy

   # Install dev dependencies with pipenv
   pipenv install --dev

   # Install adopy with flit with symlink
   pipenv run flit install -e
