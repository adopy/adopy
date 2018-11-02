.. _psi:

.. automodule:: adopy.tasks.psi

  Task
  ----

  .. autoclass:: adopy.tasks.psi.TaskPsi
    :members:

  Model
  -----

  Psychometric functions provide the probability of a subject to recognize
  the stimulus intensity. The base form of the function is as below:

  .. math::

     \Psi(x; \gamma, \lambda, \mu, \beta)
       = \gamma + (1 - \gamma - \lambda) \times F(x; \mu, \beta) \\

  where :math:`x` is the intensity of a given stimulus,
  :math:`\gamma` is the guess rate,
  :math:`\lambda` is the lapse rate,
  :math:`F(x; \mu, \beta)` is a function that defines the shape of a :math:`\Psi` function, and
  :math:`\mu` and :math:`\beta` are the threshold and the slope of the function.

  .. autoclass:: adopy.tasks.psi.ModelLogistic
    :members:
    :inherited-members:

  .. autoclass:: adopy.tasks.psi.ModelWeibull
    :members:
    :inherited-members:

  .. autoclass:: adopy.tasks.psi.ModelNormal
    :members:
    :inherited-members:

  Engine
  ------

  .. autoclass:: adopy.tasks.psi.EnginePsi
    :members:
