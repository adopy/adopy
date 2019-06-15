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

**ADOpy** is a Python implementation of adaptive design optimization (Myung, Cavagnaro, & Pitt, 2013).
Its modular design and simple structure permit easy use and
integration into existing experimentation code. Specific features include:

- Choice under risk and ambiguity task (:py:mod:`adopy.tasks.cra`)
- Delayed discounting task (:py:mod:`adopy.tasks.ddt`)
- Threshold estimation using the psi method (:py:mod:`adopy.tasks.psi`). This
  model can be used for any 2AFC task with the independent variable on a
  continuous scale.

The adaptive design optimization (ADO) consist of three steps:
(1) design optimization, (2) experimentation, and (3) Bayesian
updating. Using adopy, you can easily utilize ADO for your experimentations,
with a following style:

.. code::

    for trial in trials:
        # Compute an optimal design
        design = engine.get_design()

        # Get a response using the design (pseudo-function)
        response = get_response(design)

        # Update ADOpy engine
        engine.update(design, response)

ADOpy supports for Python 3.5 or above using NumPy, SciPy, and Pandas.

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
   :caption: API Reference

   api/base.rst
   api/tasks/*

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

