# ADOpy <img src="https://adopy.github.io/logo/adopy-logo.svg" align="right" width="300px">

[![PyPI](https://img.shields.io/pypi/v/adopy.svg?color=green)](https://pypi.org/project/adopy/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Travis CI](https://travis-ci.org/adopy/adopy.svg?branch=master)](https://travis-ci.org/adopy/adopy)
[![CodeCov](https://codecov.io/gh/adopy/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k)](https://codecov.io/gh/adopy/adopy)

**ADOpy** is a Python implementation of Adaptive Design Optimization (ADO; Myung, Cavagnaro, & Pitt, 2013), which computes optimal designs dynamically in an experiment. Its modular structure permit easy integration into existing experimentation code.

ADOpy supports Python 3.5 or above and relies on NumPy, SciPy, and Pandas.

### Features

- **Grid-based computation of optimal designs using only three classes**: `adopy.Task`, `adopy.Model`, and `adopy.Engine`.
- **Easily customizable for your own tasks and models**
- **Pre-implemented Task and Model classes including**:
  - Psychometric function estimation for 2AFC tasks (`adopy.tasks.psi`)
  - Delay discounting task (`adopy.tasks.ddt`)
  - Choice under risk and ambiguity task (`adopy.tasks.cra`)
- **Example code for experiments using PsychoPy** ([link][example-code])

[example-code]: https://github.com/adopy/adopy/tree/master/examples

### Resources

- [**Getting started**](https://adopy.org/getting-started.html)
- [**Documentation**](https://adopy.org)
- [**Bug reports**](https://github.com/adopy/adopy/issues)

## (not so) Quick-start guide ##

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
parameters (`b0`, `b1`, and `b2`):

<img src="https://user-images.githubusercontent.com/11037140/59533069-5f7b7880-8f25-11e9-8440-4d31fb6ac260.png" align="center">

How to compute the probabilty `p` should be defined as a function:

```python
import numpy as np

def calculate_prob(x1, x2, b0, b1, b2):
    """A function to compute the probability of a positive response."""
    logit = b0 + x1 * b1 + x1 * b2
    p_obs = 1. / (1 + np.exp(-logit))
    return p_obs
```

Using the information and the function, the model can be defined with
`adopy.Model`:

```python
from adopy import Model

model = Model(name='My Logistic Model',   # Name of the model (optional)
              params=['b0', 'b1', 'b2'],  # Labels of model parameters
              func=calculate_prob)        # A probability function
```

### Step 3. Define grids for design variables and model parameters

Since ADOpy uses grid search to search the design space and parameter space,
you must define a grid for design variables and model parameters.
The grid can be defined using the labels (of design variables or model
parameters) as its key and an array of the corresponding grid points
as its value.

```python
import numpy as np

grid_design = {
    'x1': np.linspace(0, 50, 100),    # 100 grid points within [0, 50]
    'x2': np.linspace(-20, 30, 100),  # 100 grid points within [-20, 30]
}

grid_param = {
    'b0': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b1': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b2': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
}
```

### Step 4. Initialize an engine using `adopy.Engine`

Using the objects created so far, an engine should be initialized using
`adopy.Engine`. It contains built-in functions to compute an optimal design
using ADO.

```python
from adopy import Engine

engine = Engine(model=model,              # a Model object
                task=task,                # a Task object
                grid_design=grid_design,  # a grid for design variables
                grid_param=grid_param)    # a grid for model parameters
```

### Step 5. Compute a design using the engine

```python
# Compute an optimal design using ADO
design = engine.get_design()
design = engine.get_design('optimal')

# Compute a randomly chosen design, as is typically done in non-ADO experiments
design = engine.get_design('random')
```

### Step 6. Collect an observation in your experiment

```python
# Get a response from a participant using your own code
response = ...
```

### Step 7. Update the engine with the observation

```python
# Update the engine with the design and the corresponding response
engine.update(design, response)
```

### Step 8. Repeat Step 5 through Step 7 until the experiment is over

```python
NUM_TRIAL = 100  # number of trials

for trial in range(NUM_TRIAL):
    # Compute an optimal design for the current trial
    design = engine.get_design('optimal')

    # Get a response using the optimal design
    response = ...  # Using users' own codes

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

