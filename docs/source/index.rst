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

**ADOpy** is a Python implementation of adaptive design optimization
[Myung2013]_. Its modular design and simple structure permit easy use and
integration into existing experimentation code. Specific features include:

- Choice under risk and ambiguity task (:py:mod:`adopy.tasks.cra`)
- Delayed discounting task (:py:mod:`adopy.tasks.ddt`)
- Threshold estimation using the psi method (:py:mod:`adopy.tasks.psi`). This
  model can be used for any 2AFC task with the independent variable on a
  continuous scale.

The adaptive design optimization (ADO) consist of three steps [Myung2013]_:
(1) design optimization, (2) experimentation, and (3) Bayesian
updating. Using adopy, you can easily utilize ADO for your experimentations,
with a following style (pseudo-code):

.. code::

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

- Python 3.6 or above
- `NumPy <http://www.numpy.org/>`_
- `SciPy <https://www.scipy.org/>`_
- `Pandas <https://pandas.pydata.org/>`_

Citation
--------

If you use ADOpy, please cite this package along with the specific version.
It greatly encourages contributors to continue supporting ADOpy.

   To be announced.

Content
-------

.. toctree::
   :maxdepth: 1
   :glob:

   install.rst
   examples/index.rst
   contributing.rst
   api/index.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

