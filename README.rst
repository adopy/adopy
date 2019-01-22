.. image:: https://user-images.githubusercontent.com/11037140/51372654-39ea6e80-1b41-11e9-86bc-fac994b9d50e.png
   :width: 300
   :align: center
   :alt: ADOpy logo

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.
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

**ADOpy** is a Python package for adaptive design optimization on experimental
tasks.
The adaptive design optimization (ADO) consist of three steps [Myung2013]_:
(1) design optimization, (2) experimentation, and (3) Bayesian updating.
Using adopy, you can utilize ADO for your experimentation, with a following
style:

.. code:: python
   :caption: Pseudo-codes for an arbitrary task using ADOpy

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

- Python 3.5+
- NumPy
- Pandas
- SciPy

Citation
--------

To be announced.

Documentation
-------------

See more details in the ADOpy documentation. Not linked yet.
