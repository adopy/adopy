Simple Example: Psi function
============================

Let’s start with an simple example: Psi function. The task has one
design variable (stimulus :math:`x`) and the models for the task have four
parameters (guess_rate :math:`\gamma`, lapse_rate :math:`\delta`,
threshold :math:`\alpha`, slope :math:`\beta`).
In this example, let’s use the logistic function for the model’s shape.
Then, the model can compute the probability of a subject to perceive the
given stimulus with the following equation:

.. math::

    p = \gamma + (1 - \gamma - \delta) \; \text{logit}^{-1}\left(
        \beta (x - \alpha)
    \right) \\
    \text{where } \text{logit}^{-1}(x) = \frac{1}{1 + e^{-x}}

In this case, let’s only assume one constraint: ``slope`` > 0.

Defining Grids for Designs and Params
-------------------------------------

To make grids for designs and parameters, you should define two dictionaries
that contains singls grids for each designs and each parameters, respectively.
In this example, we will fix the ``guess_rate`` to 0.5 and ``lapse_rate`` to 0.04.

.. code::

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

Using Predefined Classes
------------------------

To use the predefined classes for specific task and models, you can use it
with ``adopy.tasks.<task_name>``.

.. code::

   from adopy.tasks.psi import ModelLogistic, EnginePsi

   model = ModelLogistic()
   engine = EnginePsi(model=m, designs=designs, params=params)

Assuming :math:`\gamma = 0.5`, :math:`\delta = 0.04`, :math:`\alpha = 20` and :math:`\beta = 1.5`,
you can get the probability of perceiving the stimulus with ``model_log.compute``.

.. code::

   from scipy.stats import bernoulli

   # Define true parameters
   gr_true = 0.5
   lr_true = 0.04
   th_true = 20
   sl_true = 1.5

   p_obs = model_log.compute(stimulus=d_opt['stimulus'], guess_rate=gr_true, lapse_rate=lr_true,
                             threshold=th_true, slope=sl_true)
   y_obs = bernoulli.rvs(p_obs)

.. code::

   d_opt = e.get_design()

Self-defined Classes
--------------------

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

   # Define true parameters
   gr_true = 0.5
   lr_true = 0.04
   th_true = 20
   sl_true = 1.5

   p_obs = model_log.compute(stimulus=d_opt['stimulus'], guess_rate=gr_true, lapse_rate=lr_true,
                             threshold=th_true, slope=sl_true)
   y_obs = bernoulli.rvs(p_obs)

Lastly, using the optimal design and the corresponding response, the `Engine` instance can update
its posterior distributions on parameters.

.. code:: python

   engine_psi.update(d_opt, y_obs)
