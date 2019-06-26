Home
====

.. image:: https://adopy.github.io/logo/adopy-logo.svg
   :alt: ADOpy logo
   :align: center

----

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.
   :target: https://www.repostatus.org/#wip
.. image:: https://travis-ci.com/adopy/adopy.svg?token=gbyEQoyAYgexeSRwBwj6&branch=master
   :alt: Travis CI
   :target: https://travis-ci.com/adopy/adopy
.. image:: https://codecov.io/gh/adopy/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k
   :alt: CodeCov
   :target: https://codecov.io/gh/adopy/adopy

**ADOpy** is a Python implementation of Adaptive Design Optimization
(ADO; Myung, Cavagnaro, & Pitt, 2013), which computes optimal designs
dynamically in an experiment. Its modular structure permit easy integration
into existing experimentation code.

ADOpy supports Python 3.5 or above and relies on NumPy, SciPy, and Pandas.

Features
--------

- **Grid-based computation of optimal designs using only three classes**:
  :py:class:`adopy.Task`, :py:class:`adopy.Model`, and :py:class:`adopy.Engine`.
- **Easily customizable for your own tasks and models**
- **Pre-implemented Task and Model classes including**:

  - Psychometric function estimation for 2AFC tasks (:py:mod:`adopy.tasks.psi`)
  - Delay discounting task (:py:mod:`adopy.tasks.ddt`)
  - Choice under risk and ambiguity task (:py:mod:`adopy.tasks.cra`)

- **Example code for experiments using PsychoPy** (`link`_)

.. _link: https://github.com/adopy/adopy/tree/master/examples

Citation
--------

If you use ADOpy, please cite this package along with the specific version.
It greatly encourages contributors to continue supporting ADOpy.

   Yang, J., Ahn, W.-Y., Pitt., M. A., & Myung, J. I. (2019).
   *ADOpy: A Python Package for Adaptive Design Optimization*.
   Retrieved from https://adopy.org

References
----------
Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013).
A tutorial on adaptive design optimization.
*Journal of Mathematical Psychology, 57*, 53–67.

Content
-------

.. toctree::
   :maxdepth: 1
   :glob:

   getting-started.md
   dev-guide.rst

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: examples

   examples/*

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: API Reference

   api/base.rst
   api/tasks/*

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

