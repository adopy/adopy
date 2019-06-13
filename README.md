# ADOpy <img src="https://adopy.github.io/logo/adopy-logo.svg" align="right" width="300px">

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Travid CI](https://travis-ci.com/adopy/adopy.svg?token=gbyEQoyAYgexeSRwBwj6&branch=master)](https://travis-ci.com/adopy/adopy)
[![CodeCov](https://codecov.io/gh/adopy/adopy/branch/master/graph/badge.svg?token=jFnJgnVV1k)](https://codecov.io/gh/adopy/adopy)

**ADOpy** is a Python package for the Adaptive Design Optimization to compute optimal designs dynamically in an experiment (Myung, Cavagnaro, & Pitt, 2013).
Its modular design and simple structure permit easy use and integration into existing experimentation code.

It supports for Python 3.5 or above and largely based on NumPy, SciPy, and Pandas.

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

### Step 1. Define a task

```python
from adopy import Task

task = Task(name='My New Experiment',  # Name of the task
            designs = ['x1', 'x2'],    # Labels of design variables
            responses = [0, 1])        # Possible responses
```

### Step 2. Define a model

```python
import numpy as np

def calculate_likelihood(x1, x2, b0, b1, b2):
    """A function to compute likelihood for getting a positive response."""
    logit = b0 + x1 * param1 + x1 * b2
    p_obs = 1. / (1 + np.exp(-logit))
    return p_obs
```

```python
from adopy import Model

model = Model(name='My Logistic Model',   # Name of the model
              params=['b0', 'b1', 'b2'],  # Labels of model parameters
              func=calculate_likelihood)  # A function to compute likelihood
```

### Step 3. Define grids for design variables and model parameters

```python
import numpy as np

designs = {
    'x1': np.linspace(0, 50, 100),    # 100 grid points within [0, 50]
    'x2': np.linspace(-20, 30, 100),  # 100 grid points within [-20, 30]
}

params = {
    'b0': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b1': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    'b2': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
}
```

### Step 4. Initialize an engine

```python
from adopy import Engine

engine = Engine(model=model,      # a Model object
                task=task,        # a Task object
                designs=designs,  # a grid for design variables
                params=params)    # a grid for model parameters
```

### Step 5. Compute a design using the engine

```python
# Compute an optimal design using the Adaptive Design Optimization
design = engine.get_design('optimal')

# Or compute a randomly chosen design
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

    # Returns a binary response using Bernoulli distribution
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

> Yang, J., Ahn, W.-Y., Pitt., M. A., & Myung, J. I. (in preparation). *ADOpy: A Python Package for Adaptive Design Optimization*. Retrieved from https://adopy.org

## References

- Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013).
  A tutorial on adaptive design optimization.
  *Journal of Mathematical Psychology, 57*, 53–67.
