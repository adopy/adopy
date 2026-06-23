# ADOpy Agent Instructions

## What is not obvious from the code

ADOpy is a scientific package for Adaptive Design Optimization (ADO) in
behavioral experiments. Treat code changes as changes to experimental design,
not as ordinary data-processing refactors. Small numerical edits can alter trial
selection, posterior estimates, and published reproducibility.

ADOpy's responsibility stops at design selection and Bayesian updating. It does
not present stimuli, collect participant responses, save experiment data, or
replace PsychoPy/OpenSesame/Expyriment-style experiment runtimes unless the user
explicitly asks for an integration layer.

The intended experiment loop is:

1. Define a `Task`.
2. Define a `Model`.
3. Define design, parameter, and response grids.
4. Initialize an `Engine`.
5. Call `engine.get_design(...)`.
6. Collect a participant or simulated response outside ADOpy.
7. Call `engine.update(design, response)`.
8. Repeat.

## Local harness

Project-local reusable skills live under `.agents/skills/`:

- `.agents/skills/adopy-core-numerics/SKILL.md`
- `.agents/skills/adopy-task-model/SKILL.md`
- `.agents/skills/adopy-docs-release/SKILL.md`

Use the relevant skill before changing its target area.

## API contract

Preserve Python `>=3.6` compatibility unless the task explicitly authorizes
modernization.

Key non-obvious behavior to preserve:

- `Task.responses` stores response-variable labels, not possible response
  values.
- Possible response values belong in `Engine(..., grid_response=...)`.
- `Model.func` and `Model.compute()` return log likelihoods, not binary-response
  probabilities.
- `Model.func` argument names must include every task design label, response
  label, and model parameter label.
- `Engine.update()` accepts one design/response pair or matched lists of pairs.
- `Engine(dtype=...)` defaults to `numpy.float32`; changing dtype behavior is a
  behavioral change.

Do not add deprecated aliases, compatibility shims, or parallel APIs unless
explicitly requested. Prefer clean migration of docs, tests, and callsites.

## Numerical guardrails

Be especially conservative around:

- likelihood/log-likelihood semantics
- posterior normalization
- `scipy.special.logsumexp`
- entropy, conditional entropy, marginal entropy, and mutual information
- cached `Engine` fields beginning with `_log_lik`, `_marg_log_lik`, `_ent`,
  `_ent_marg`, `_ent_cond`, or `_mutual_info`
- grid construction and tuple-key multi-column grids
- nearest-grid matching
- random design paths
- `noise_ratio`
- dtype and precision
- memory allocation from Cartesian grids or precomputed lookup tables

Grid-based ADO has exponential memory pressure as design, parameter, or response
dimensions grow. Do not casually increase default grid sizes or dimensionality.

## Scientific references

Use the project docs and papers as the conceptual anchor for scientific changes:

- ADOpy docs: https://adopy.github.io/adopy/
- ADOpy package paper: Yang, Pitt, Ahn, & Myung, Behavior Research Methods
  53(2):874-897, DOI `10.3758/s13428-020-01386-4`
- ADO tutorial: Myung, Cavagnaro, & Pitt, Journal of Mathematical Psychology
  57:53-67, DOI `10.1016/j.jmp.2013.05.005`

For task-specific models, keep terminology, parameter names, equations, and
citations aligned with module docstrings and docs.

## Verification expectations

Run the narrowest meaningful verification for changed files:

- `adopy/base/*` -> `uv run --extra test pytest tests/test_base.py`
- `adopy/functions/*` -> `uv run --extra test pytest tests/test_functions.py`
- `adopy/tasks/psi.py` -> `uv run --extra test pytest tests/test_psi.py`
- `adopy/tasks/dd.py` -> `uv run --extra test pytest tests/test_dd.py`
- `adopy/tasks/cra.py` -> `uv run --extra test pytest tests/test_cra.py`
- Docs/API changes -> build or inspect the affected Sphinx page and verify
  current API names.
- Packaging/release changes -> verify version, changelog, README/docs, and
  package metadata surfaces together.

Prefer small deterministic tests over broad stochastic assertions. For random
paths, assert invariants or seed the RNG.
