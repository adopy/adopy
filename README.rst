.. raw:: html

   <div align="center">
     <img src="https://adopy.github.io/logo/adopy-logo.svg">
   </div> <!-- ADOpy logo -->

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Project Status: WIP – Initial development is in progress,
         but there has not yet been a stable, usable release suitable for the public.
   :target: https://www.repostatus.org/#wip
.. image:: https://travis-ci.com/JaeyeongYang/adopy.svg?token=gbyEQoyAYgexeSRwBwj6&branch=master
   :alt: Travis CI
   :target: https://travis-ci.com/JaeyeongYang/adopy
.. image:: https://codecov.io/gh/JaeyeongYang/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k
   :alt: CodeCov
   :target: https://codecov.io/gh/JaeyeongYang/adopy
.. image:: https://www.codefactor.io/repository/github/jaeyeongyang/adopy/badge
   :alt: CodeFactor
   :target: https://www.codefactor.io/repository/github/jaeyeongyang/adopy

**ADOpy** is a Python implementation of adaptive design optimization [Myung2013]_ Its modular design and 
simple structure permit easy use and integration into existing experimentation code. Specific
features include:

- Threshold estimation using the psi method (). This model can be used for any 2AFC task with the independent variable on a continuous scale.
-
-
-
on experimental
tasks. The adaptive design optimization (ADO) consist of three steps
[Myung2013]_: (1) design optimization, (2) experimentation, and (3) Bayesian
updating. Using adopy, you can easily utilize ADO for your experimentations,
with a following style (pseudo-code):

.. code:: python

   for trial in trials:
       design = engine.get_design()
       response = get_response(design)
       engine.update(design, response)

.. [Myung2013]
   Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013).
   A tutorial on adaptive design optimization.
   *Journal of Mathematical Psychology, 57*, 53–67.

Dependencies
------------

- Python 3.5+ (no support for Python 2)
- `NumPy <http://www.numpy.org/>`_
- `SciPy <https://www.scipy.org/>`_
- `Pandas <https://pandas.pydata.org/>`_

Citation
--------

If you use ADOpy, please cite this package along with a specific version. It greatly encourages  contributors to
continue supporting ADOpy.

   To be announced.

Documentation
-------------

See more details in the ADOpy documentation. Not linked yet.

* Home

  * Installation
  * Examples
  * Contributing
  * API References
