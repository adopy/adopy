Changelogs
==========

0.4.0
-----

* Drop support for Python 3.5.
* **(breaking change)** :py:mod:`adopy.base.Task` takes labels of response variables
  as the argument `responses`. Instead, the possible values for the response
  variables should be given to :py:mod:`adopy.base.Engine` as an argument
  named `grid_response`.
* **(breaking change)** :py:mod:`adopy.base.Model` takes a log likelihood function
  for an argument `func` now, instead of the probability function for a single
  binary response variable. The log likelihood function can take multiple
  response variables.
* **(breaking change)** The `compute` method in :py:mod:`adopy.base.Model`
  provides the log likelihood function now, instead of the probability of a
  single binary response variable.
* Using multiple response variables is available now!
* :py:mod:`adopy.base.Engine` now can update multiple observations, given as a
  list of designs and a list of corresponding responses into :code:`design` and
  :code:`response` arguments, respectively.
* Now, you can choose what datatype to use for :py:mod:`adopy.base.Engine`,
  with an argument named `dtype`. The default is `numpy.float32`.
* Remove unusing types at base (:issue:`26`; contributed by :user:`NicholasWon47`)

0.3.1
-----

A minor update due to PyPI configuration. It has the exact same
features as the previous version.

0.3.0
-----

This is the first version released publicly. It includes following modules:

* Base classes (:py:mod:`adopy.base`)
* Choice under risk and ambiguity task (:py:mod:`adopy.tasks.cra`)
* Delay discounting task (:py:mod:`adopy.tasks.dd`)
* Psychometric function estimation (:py:mod:`adopy.tasks.psi`)

