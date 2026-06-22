---
name: adopy-task-model
description: Use when adding or modifying preimplemented ADOpy task, model, or task-specific engine modules under adopy/tasks.
---

# ADOpy Task and Model Modules

Use this skill before editing `adopy/tasks/*.py` or adding a new preimplemented experimental task.

## Existing task modules

- `adopy.tasks.psi`: psychometric function estimation / 2AFC.
- `adopy.tasks.dd`: delay discounting.
- `adopy.tasks.cra`: choice under risk and ambiguity.

Each module should keep the same triad:

1. `Task*` class with design and response labels.
2. One or more `Model*` classes with named model parameters and log-likelihood computation.
3. `Engine*` wrapper that supplies task-specific defaults and validates model/task compatibility.

## Implementation rules

- Match design labels, response labels, parameter labels, likelihood-function arguments, grid keys, tests, and docs exactly.
- Return log likelihoods from model computations. Use `scipy.stats.*.logpmf` or equivalent log-space computation.
- Keep model names, parameter names, and equations aligned with source literature and module docstrings.
- Do not change task defaults, response coding, or parameter semantics without updating tests and docs.
- Do not expand ADOpy into participant-presentation or response-capture code. Keep experiment runtime code outside the package unless explicitly requested.
- Keep grids small in tests. Tests should prove behavior and compatibility, not run publication-scale simulations.

## Adding a new task

A complete new task requires:

- a new module under `adopy/tasks/`
- a `Task*` class
- at least one `Model*` class
- an `Engine*` wrapper or explicit reason the base `Engine` is enough
- a pytest file covering initialization, `get_design`, and `update`
- a Sphinx API page under `docs/source/api/tasks/`
- an entry in the docs index if users should discover it
- citations or source notes for the experimental task and model equations

## Verification

Run the task-specific test file:

```bash
uv run --extra test pytest tests/test_psi.py
uv run --extra test pytest tests/test_dd.py
uv run --extra test pytest tests/test_cra.py
```

For a new task, add and run the corresponding new test file. If core engine assumptions change, also run `tests/test_base.py`.
