# ADOpy <img src="https://adopy.github.io/logo/adopy-logo.svg" align="right" width="300px">

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Travid CI](https://travis-ci.com/adopy/adopy.svg?token=gbyEQoyAYgexeSRwBwj6&branch=master)](https://travis-ci.com/adopy/adopy)
[![CodeCov](https://codecov.io/gh/adopy/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k)](https://codecov.io/gh/adopy/adopy)

**ADOpy** is a Python package for the Adaptive Design Optimization (ADO; Myung, Cavagnaro, & Pitt, 2013) to compute optimal designs dynamically in an experiment.
Its modular design and simple structure permit easy use and integration into existing experimentation code.

It provides specific features:

- **Three basic classes to compute Grid-based ADO**: `adopy.Task`, `adopy.Model`, and `adopy.Engine`.
- **Customizable for your own tasks and models**
- **Pre-implemented Task and Model classes including**:
  - Psychometric function estimation (`adopy.tasks.psi`)
  - Delay discounting task (`adopy.tasks.ddt`)
  - Choice under risk and ambiguity task (`adopy.tasks.cra`)
- **[Example codes][example-code] for experiments using PsychoPy**

[example-code]: https://github.com/adopy/adopy/tree/master/examples

ADOpy supports for Python 3.5 or above and largely based on NumPy, SciPy, and Pandas.

- [**Getting started**](https://adopy.org/getting-started.html)
- [**Documentation**](https://adopy.org)
- [**Bug reports**](https://github.com/adopy/adopy/issues)

## Quick-start guides

### Step 0. Install ADOpy on the terminal

```bash
# Install the stable version from PyPI
pip install adopy

# Or install the developmental version from GitHub
git clone https://github.com/adopy/adopy.git
cd adopy
git checkout develop
pip install .
```

### Step 1. Define a task using `adopy.Task`

Assume that a user want to use ADOpy for an *arbitrary* task with two design
variables (`x1` and `x2`) where participants can make a binary choice on each
trial. Then, the task can be defined with `adopy.Task` as described below:

```python
from adopy import Task

task = Task(name='My New Experiment',  # Name of the task (optional)
            designs = ['x1', 'x2'],    # Labels of design variables
            responses = [0, 1])        # Possible responses
```

### Step 2. Define a model using `adopy.Model`

To predict partipants' choices, here we assume a logistic regression model
that calculates the probability to make a positive response using three model
parameters (`b0`, `b1`, and `b2`) as an equation below:

<img src="https://user-images.githubusercontent.com/11037140/59533069-5f7b7880-8f25-11e9-8440-4d31fb6ac260.png" align="center">

Then, how to compute the probabilty should be defined as a function:

```python
import numpy as np

def calculate_prob(x1, x2, b0, b1, b2):
    """A function to compute the probability of a positive response."""
    logit = b0 + x1 * b1 + x1 * b2
    p_obs = 1. / (1 + np.exp(-logit))
    return p_obs
```

Using the information and the function, the model can be defined with
`adopy.Model` as described below:

```python
from adopy import Model

model = Model(name='My Logistic Model',   # Name of the model (optional)
              params=['b0', 'b1', 'b2'],  # Labels of model parameters
              func=calculate_prob)        # A probability function
```

### Step 3. Define grids for design variables and model parameters

Since ADOpy uses grid search for the design space and parameter space,
you should define a grid for design variables and model parameters.
The grid can be defined using the labels (of design variables or model
parameters) as its key and an array of the corresponding grid points
as its value.

```python
import numpy as np

grid_designs = {
    'x1': np.linspace(0, 50, 100),    # 100 grid points within [0, 50]
    'x2': np.linspace(-20, 30, 100),  # 100 grid points within [-20, 30]
}

grid_params = {
    'b0': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b1': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b2': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
}
```

To make constraints on design variables, you should pass a joint matrix
of which each column corresponds to a grid point of a design variable.
Then, the key on the grid object should be a list of design variables
with the same order as in the columns of the joint matrix.

```python
# Define a joint matrix with a constraint, x1 > x2.
x_joint = []
for x1 in np.linspace(0, 50, 101):        # 101 grid points within [0, 50]
    for x2 in np.linspace(-20, 30, 101):  # 101 grid points within [-20, 30]
        if x1 > x2:
            x_joint.append([x1, x2])
#   x1   x2
# [[0, -20  ],
#  [0, -19.5],
#  ...,
#  [50, 29.5],
#  [50, 30  ]]

grid_designs = {
    ('x1', 'x2'): x_joint
}
```

### Step 4. Initialize an engine using `adopy.Engine`

Using the objects created so far, an engine should be initialized using
`adopy.Engine`. It contains built-in functions to compute an optimal design
based on the Adaptive Design Optimization.

```python
from adopy import Engine

engine = Engine(model=model,           # a Model object
                task=task,             # a Task object
                designs=grid_designs,  # a grid for design variables
                params=grid_params)    # a grid for model parameters
```

### Step 5. Compute a design using the engine

```python
# Compute an optimal design based on the ADO
design = engine.get_design()
design = engine.get_design('optimal')

# Compute a randomly chosen design
design = engine.get_design('random')
```

### Step 6. Run an experiment using the design

```python
# Get a response from a real experiment using your own codes,
response = ...


# Or simulate a response using the model object.
from scipy.stats import bernoulli

def get_simulated_response(model, design):
    """Simulate a response using b0 = 1.2, b1 = 3.7 and b2 = -2.5."""
    # Compute the likelihood to get a positive response of 1.
    p_obs = model.compute(x1=design['x1'], x2=design['x2'], b0=1.2, b1=3.7, b2=-2.5)

    # Simulate a binary choice response using Bernoulli distribution
    return bernoulli.rvs(p_obs)

response = get_simulated_response(model, design)
```

### Step 7. Update the engine from the observation

```python
# Update the engine with the design and the corresponding response
engine.update(design, response)
```

### Step 8. Repeat from Step 5 to Step 7 until the end

```python
NUM_TRIAL = 100  # number of trials

for trial in range(NUM_TRIAL):
    # Compute an optimal design for the current trial
    design = engine.get_design('optimal')

    # Get a simulated response
    response = get_simulated_response(model, design)

    # Update the engine
    engine.update(design, response)
```

## Citation

If you use ADOpy, please cite this package along with the specific version.
It greatly encourages contributors to continue supporting ADOpy.

> Yang, J., Ahn, W.-Y., Pitt., M. A., & Myung, J. I. (2019).
> *ADOpy: A Python Package for Adaptive Design Optimization*.
> Retrieved from https://adopy.org

## References

- Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013).
  A tutorial on adaptive design optimization.
  *Journal of Mathematical Psychology, 57*, 53–67.
