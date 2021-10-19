# ADOpy <img src="https://adopy.github.io/logo/adopy-logo.svg" align="right" width="300px">

[![PyPI](https://img.shields.io/pypi/v/adopy.svg?color=green)](https://pypi.org/project/adopy/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Travis CI](https://travis-ci.org/adopy/adopy.svg?branch=develop)](https://travis-ci.org/adopy/adopy)
[![CodeCov](https://codecov.io/gh/adopy/adopy/branch/develop/graph/badge.svg?token=jFnJgnVV1k)](https://codecov.io/gh/adopy/adopy)

**ADOpy** is a Python implementation of Adaptive Design Optimization (ADO; Myung, Cavagnaro, & Pitt, 2013), which computes optimal designs dynamically in an experiment. Its modular structure permit easy integration into existing experimentation code.

ADOpy supports Python 3.6 or above and relies on NumPy, SciPy, and Pandas.

### Features

- **Grid-based computation of optimal designs using only three classes**: `adopy.Task`, `adopy.Model`, and `adopy.Engine`.
- **Easily customizable for your own tasks and models**
- **Pre-implemented Task and Model classes including**:
  - Psychometric function estimation for 2AFC tasks (`adopy.tasks.psi`)
  - Delay discounting task (`adopy.tasks.ddt`)
  - Choice under risk and ambiguity task (`adopy.tasks.cra`)
- **Example code for experiments using PsychoPy** ([link][example-code])

[example-code]: https://github.com/adopy/adopy/tree/master/examples

### Installation

```bash
# Install from PyPI
pip install adopy

# Install from Github (developmental version)
pip install git+https://github.com/adopy/adopy.git@develop
```

### Resources

- [**Getting started**](https://adopy.org/getting-started.html)
- [**Documentation**](https://adopy.org)
- [**Bug reports**](https://github.com/adopy/adopy/issues)

## Citation

If you use ADOpy, please cite this package along with the specific version.
It greatly encourages contributors to continue supporting ADOpy.

> Yang, J., Pitt, M. A., Ahn, W., & Myung, J. I. (2020).
> ADOpy: A Python Package for Adaptive Design Optimization.
> _Behavior Research Methods_, 1-24.
> https://doi.org/10.3758/s13428-020-01386-4

## Acknowledgement

The research was supported by National Institute of Health Grant R01-MH093838 to Mark A. Pitt and Jay I. Myung, the Basic Science Research Program through the National Research Foundation (NRF) of Korea funded by the Ministry of Science, ICT, & Future Planning (NRF-2018R1C1B3007313 and NRF-2018R1A4A1025891), the Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01367, BabyMind), and the Creative-Pioneering Researchers Program through Seoul National University to Woo-Young Ahn.

## References

- Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013).
  A tutorial on adaptive design optimization.
  _Journal of Mathematical Psychology, 57_, 53–67.
