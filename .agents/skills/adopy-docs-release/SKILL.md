---
name: adopy-docs-release
description: Use when changing ADOpy README, Sphinx docs, examples, changelog, version metadata, packaging, CI, or release instructions.
---

# ADOpy Docs and Release Surface

Use this skill before editing documentation, release, or packaging surfaces such as:

- `README.md`
- Sphinx docs and examples
- package metadata and manifests
- CI, Read the Docs, changelog, version, or release instructions

Inspect the repository for the current filenames before editing. This skill should capture durable documentation/release rules, not a live inventory of every page, config file, or temporary drift item.

## Documentation rules

- Use current post-0.4.0 API names: `responses`, `params`, `grid_design`, `grid_param`, and `grid_response`.
- Avoid deprecated example arguments such as `designs=`, `params=`, `param=`, `design=`, or `y_obs=` unless documenting migration history.
- Verify module names against source before documenting imports.
- Keep citations and package-version guidance visible for scientific changes.
- If docs describe a model equation or task workflow, check that source code and tests still match.

## Release/change checklist

For user-facing behavior changes, update together:

- code docstrings
- Sphinx API/example docs
- tests
- changelog
- README if the public feature list or install/use pattern changes
- version metadata if preparing a release

For packaging changes, verify source distribution inclusion/exclusion and docs build requirements.

## Verification

Use the narrowest relevant checks:

```bash
uv run --extra test pytest tests
cd docs && uv run --extra docs make html
```

If a full docs build is not practical, inspect the affected `.rst`/notebook source and the corresponding Python API signatures, then state the limitation explicitly.
