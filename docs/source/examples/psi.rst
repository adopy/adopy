Psychometric Function Estimation
================================

Let’s start with psychometric functions as an example. The goal of the function
is to figure out whether a subject can perceive a signal with varying levels
of magnitude. The function has one design variable for the *intensity* of a
stimulus, :math:`x`; the model has four model parameters:
*guess rate* :math:`\gamma`, *lapse rate* :math:`\delta`,
*threshold* :math:`\alpha`, and *slope* :math:`\beta`.

.. figure:: ../_static/images/Psychometricfn.svg
   :width: 70%
   :align: center

   A simple diagram for the Psychometric function.

In this example, let’s use the **logistic function** for the model’s shape.
Then, the model can compute the probability of a subject to perceive the
given stimulus with the following equation:

.. math::

   \Psi(x \mid \alpha, \beta, \gamma, \delta)
   = \gamma + (1 - \gamma - \delta) \; \sigma\big( \beta (x - \alpha) \big)
   \quad \text{where } \sigma(x) = \frac{1}{1 + e^{-x}}

For this example, let's assume the true parameters as :math:`\gamma = 0.5`,
:math:`\delta = 0.04`, :math:`\alpha = 20`, and :math:`\beta = 1.5`.

.. code:: python

  # Define true parameters
  GR_TRUE = 0.5
  LR_TRUE = 0.04
  TH_TRUE = 20
  SL_TRUE = 1.5

Preparing grids
---------------

To make grids for designs and parameters, you should define two dictionaries
that contain singles grids for all designs and all parameters, respectively.
In this example, we will fix the ``guess_rate`` to 0.5 and ``lapse_rate`` to 0.04.

.. code:: python

  import numpy as np

  designs = {
      'stimulus': np.linspace(20 * np.log10(.05), 20 * np.log10(400), 120)
  }

  params = {
      'guess_rate': [0.5],
      'lapse_rate': [0.04],
      'threshold': np.linspace(20 * np.log10(.1), 20 * np.log10(200), 200),
      'slope': np.linspace(0, 10, 200)
  }

Using pre-defined classes
-------------------------

To use the predefined classes for specific task and models, you can use it
with `adopy.tasks.<task_name>`, e.g., ``adopy.tasks.psi``.

.. code:: python

  from adopy.tasks.psi import ModelLogistic, EnginePsi

  model = ModelLogistic()
  engine = EnginePsi(model=model, designs=designs, params=params)

Using `compute()` method of the model instance, you can compute the probability
for a subject to succeed to perceive a signal.

.. code:: python

  model.compute(stimulus=10, guess_rate=0.5, lapse_rate=0.04,
                threshold=10, slope=0.5)

.. code:: python

  from scipy.stats import bernoulli

  p_obs = model.compute(stimulus=d_opt['stimulus'],
                        guess_rate=gr_true, lapse_rate=lr_true,
                        threshold=th_true, slope=sl_true)
  y_obs = bernoulli.rvs(p_obs)

.. code:: python

  d_opt = e.get_design()

Using self-defined classes
--------------------------

Instead of using pre-defined classes, they can be implemented as ``Task`` and ``Model`` objects by the
codes below:

.. code:: python

  import numpy as np
  from adopy import Task, Model

  task_psi = Task(name='Psi', key='psi', design=['stimulus'])


  def inv_logit(x):
      return np.divide(1, 1 + np.exp(-x))

  def func_logistic(stimulus, guess_rate, lapse_rate, threshold, slope):
      return guess_rate + (1 - guess_rate - lapse_rate) * inv_logit(slope * (stimulus - threshold))


  model_log = Model(name='Logistic', key='log', task=task_psi,
                    param=['guess_rate', 'lapse_rate', 'threshold', 'slope'],
                    func=func_logistic, constraint={'slope': lambda x: x > 0})

Then, you can compute the probability using ``compute`` method in the
model object.

.. code:: python

  print(model_log.compute(stimulus=10, guess_rate=0.5, lapse_rate=0.04, threshold=10, slope=0.5))
  print(model_log.compute(stimulus=15, guess_rate=0.5, lapse_rate=0.04, threshold=10, slope=0.5))
  print(model_log.compute(stimulus=5, guess_rate=0.5, lapse_rate=0.04, threshold=10, slope=0.5))

Now, if you want to use an ADO engine for the task and the model,

.. code:: python

  from adopy import Engine

  engine_psi = Engine(task=task_psi, model=model_log,
                      designs=designs, params=params, y_obs=[0, 1])

With the ``Engine`` instance, you can get the optimal design:

.. code:: python

  d_opt = engine_psi.get_design()

Assuming :math:`\gamma = 0.5`, :math:`\delta = 0.04`, :math:`\alpha = 20` and :math:`\beta = 1.5`,
you can get the probability of perceiving the stimulus with `model_log.compute`.

.. code:: python

  from scipy.stats import bernoulli

  p_obs = model_log.compute(stimulus=d_opt['stimulus'], guess_rate=gr_true, lapse_rate=lr_true,
                            threshold=th_true, slope=sl_true)
  y_obs = bernoulli.rvs(p_obs)

Lastly, using the optimal design and the corresponding response, the `Engine` instance can update
its posterior distributions on parameters.

.. code:: python

  engine_psi.update(d_opt, y_obs)
