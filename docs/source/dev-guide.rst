Developer's Guide
=================

Setup
-----

First, you should clone the GitHub repository and checkout the develop branch.

.. code:: bash

    git clone git@github.com:adopy/adopy.git
    cd adopy
    git checkout develop

To manage a virtual environment for development, we use `uv`_. The project
metadata and dependency groups are defined in ``pyproject.toml`` and locked in
``uv.lock``.

.. _uv:
   https://docs.astral.sh/uv/

.. code:: bash

    # Install runtime dependencies, development tools, and optional docs/test extras
    uv sync --all-extras

Run project commands through the synchronized environment:

.. code:: bash

    uv run --extra test pytest

Writing documentation
---------------------

The documentation is generated based on `Sphinx`_. Sphinx provides its rich
features using `reStructuredText`_ files for its markup language. Thus, to get
benefits from Sphinx, this documentation website is created from
reStructuredText sources on ``/docs/source`` directory.

.. _Sphinx:
   http://www.sphinx-doc.org/en/master/
.. _reStructuredText:
   http://docutils.sourceforge.net/docs/user/rst/quickstart.html

The official website (https://adopy.org) is generated from documentation in
the Python code under ``/adopy`` and reStructuredText files under
``/docs/source``. Keep source docstrings and Sphinx pages in sync when changing
public APIs.

Using ``sphinx-autobuild``, you can build the documentation and test it by
yourself. It can run a web server for documentation on http://localhost:8000,
as described below.

.. _sphinx-autobuild:
   https://pypi.org/project/sphinx-autobuild/

.. code:: bash

    # Build the documentation once
    uv run --extra docs sphinx-build -b html docs/source docs/build/html

    # Or run a local live-reload server from the docs directory
    cd docs
    uv run --extra docs make livehtml

Branch management
-----------------

ADOpy uses ``develop`` for ongoing development and ``release/*`` branches for
release preparation. Feature work should use focused ``feature/*`` or
``fix/*`` branches and open pull requests against the branch they build on.
Open a GitHub issue before larger changes so the task and review scope are
clear.

If you want to contribute to ADOpy, start by forking the ADOpy repository on
GitHub. Before writing code, create or reference an issue for the work. Use a
focused branch, open a pull request against the appropriate base branch, and
wait for review before merging.

