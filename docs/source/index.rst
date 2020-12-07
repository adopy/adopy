Home
====

.. image:: https://adopy.github.io/logo/adopy-logo.svg
   :alt: ADOpy logo
   :align: center

----

.. image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. image:: https://travis-ci.org/adopy/adopy.svg?branch=develop
   :alt: Travis CI
   :target: https://travis-ci.org/adopy/adopy
.. image:: https://codecov.io/gh/adopy/adopy/branch/develop/graph/badge.svg?token=jFnJgnVV1k
   :alt: CodeCov
   :target: https://codecov.io/gh/adopy/adopy

**ADOpy** is a Python implementation of Adaptive Design Optimization
(ADO; Myung, Cavagnaro, & Pitt, 2013), which computes optimal designs
dynamically in an experiment. Its modular structure permit easy integration
into existing experimentation code.

ADOpy supports Python 3.6 or above and relies on NumPy, SciPy, and Pandas.

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

   Yang, J., Pitt, M. A., Ahn, W., & Myung, J. I. (2020).
   ADOpy: A Python Package for Adaptive Design Optimization.
   *Behavior Research Methods*, 1--24.
   https://doi.org/10.3758/s13428-020-01386-4

Acknowledgement
---------------

The research was supported by National Institute of Health Grant R01-MH093838
to Mark A. Pitt and Jay I. Myung, the Basic Science Research Program through
the National Research Foundation (NRF) of Korea funded by the Ministry of Science,
ICT, & Future Planning (NRF-2018R1C1B3007313 and NRF-2018R1A4A1025891),
the Institute for Information & Communications Technology Planning & Evaluation
(IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01367, BabyMind),
and the Creative-Pioneering Researchers Program through Seoul National University
to Woo-Young Ahn.

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
   changelogs.rst

.. toctree::
   :maxdepth: 3
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

