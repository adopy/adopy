# TODOs

Temporary holding area for repository drift found while adding project agent support.

## Documentation/API drift

- Replace stale `adopy.tasks.ddt` mentions with implemented module name `adopy.tasks.dd` in README/docs.
- Rewrite `docs/source/examples/psi.rst` to use the post-0.4 API (`responses`, `params`, `grid_design`, `grid_param`, `grid_response`) instead of deprecated example arguments.
- Audit `docs/source/getting-started.rst` logistic example; `logit = b0 + x1 * b1 + x1 * b2` likely should use `x2 * b2` for the second design variable.

## Release/package drift

- Align version surfaces across local metadata, docs, package fallback version, and PyPI release notes.
- Review `MANIFEST.in`; it includes `README.rst` even though the repository uses `README.md`.
- Review old Travis/Gitflow/master-branch documentation against the current default branch and CI/deploy reality.
