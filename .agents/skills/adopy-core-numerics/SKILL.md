---
name: adopy-core-numerics
description: Use when changing ADOpy core Task/Model/Engine behavior, grid utilities, posterior updates, entropy, mutual information, dtype, random design selection, or numerical performance.
---

# ADOpy Core Numerics

Use this skill before editing:

- `adopy/base/_task.py`
- `adopy/base/_model.py`
- `adopy/base/_engine.py`
- `adopy/functions/_grid.py`
- `adopy/functions/_utils.py`
- `adopy/functions/_const.py`
- `adopy/types.py`

## Contract to preserve

- `Task` stores labels for design variables and response variables.
- `Model` stores a `Task`, parameter labels, and an optional log-likelihood function.
- `Model.compute()` returns log likelihood. It must not silently return probability-space values for real models.
- `Engine` builds design, parameter, and response grids; computes likelihood tables; updates log posterior; and chooses designs by mutual information.
- Posterior updates must remain normalized in log space.
- `dtype` defaults to `numpy.float32`.

## Workflow

1. Identify the exact mathematical or API invariant being changed.
2. Inspect current tests and docs for that invariant before editing.
3. Keep computations vectorized with NumPy/Pandas/SciPy; avoid Python loops over full grid products unless the grid is explicitly tiny.
4. Keep probability-sensitive operations in log space. Use `logsumexp` where normalization or marginalization requires it.
5. Check memory shape before adding or expanding lookup tables. Grid-based ADO scales poorly with extra dimensions.
6. Add or update the narrow test that would fail for the bug or behavior change.

## Red flags

Stop and re-check the math if a change touches:

- `self._log_lik`, `self._marg_log_lik`, `self._ent`, `self._ent_marg`, `self._ent_cond`, or `self._mutual_info`
- `noise_ratio`
- response-grid indexing
- tuple-key grids in `make_grid_matrix`
- random selection from `np.random`
- conversion between DataFrame, Series, ndarray, dict, and OrderedDict
- dtype casts

## Verification

Run the narrowest relevant command:

```bash
uv run --extra test pytest tests/test_base.py
uv run --extra test pytest tests/test_functions.py
```

If behavior depends on a task module, also run that task's test file. For random paths, seed or assert invariants instead of exact sampled values.
