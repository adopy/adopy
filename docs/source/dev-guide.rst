Developer's Guide
=================

Setup
-----

First, you should clone the GitHub repository and checkout the develop branch.

.. code:: bash

    git clone git@github.com:adopy/adopy.git
    cd adopy
    git checkout develop

To manage a virtual environment for development, we uses `Pipenv`_. Pipfile on
the repository describes which packages to install for a virtual environment.
You can create a virtual environment by running the command below:

.. _Pipenv:
   https://docs.pipenv.org/en/latest/

.. code:: bash

    # Install default and develop packages while skipping Pipfile.lock
    pipenv install --dev --skip-lock

Now you can activate the installed virtual environment. Code blocks on later
sections assume this environment activated.

.. code:: bash

    # Activate the virtual environment
    pipenv shell

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

Basically, the official website (https://adopy.org) is automatically generated
from documentations in the Python codes on ``/adopy`` and reStructuredText
files (``*.rst``) on ``/docs/source``, in ``master`` branch of the
repository. In other words, the documentations shown in the website are for the
stable version of ADOpy in ``master`` branch.

Using ``sphinx-autobuild``, you can build the documentation and test it by
yourself. It can run a web server for documentation on http://localhost:8000,
as described below.

.. _sphinx-autobuild:
   https://pypi.org/project/sphinx-autobuild/

.. code:: bash

    # Go to /docs directory
    cd docs

    # On Windows
    livehtml.bat

    # On macOS or Linux
    make livehtml

Branch management
-----------------

The basic process of ADOpy development follows the `Gitflow workflow by Vincent
Driessen`_, which holds two main branches: ``master`` and ``develop``.
Only an exception is that we use ``feature/*``, ``hotfix/*``, or ``release/*``
instead of ``feature-*``, ``hotfix-*`` or ``release-*``.
There are a bunch of great resources about Gitflow and we require you to read
at least one of them before proceeding.

* `The original post by Vincent Driessen`_
* `Gitflow Workflow - Atlassian Git Tutorial`_
* `Read Git Flow | Leanpub`_
* `git-flow cheatsheet`_
* `Managing your Git branches with Git Flow | Zell Liew`_
* `Using git-flow to automate your git branching workflow`_


.. _Gitflow workflow by Vincent Driessen:
   https://nvie.com/posts/a-successful-git-branching-model/
.. _The original post by Vincent Driessen:
   https://nvie.com/posts/a-successful-git-branching-model/
.. _Gitflow Workflow - Atlassian Git Tutorial:
   https://ko.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
.. _Read Git Flow | Leanpub:
   https://leanpub.com/git-flow/read
.. _git-flow cheatsheet:
   https://danielkummer.github.io/git-flow-cheatsheet/index.html
.. _Managing your Git branches with Git Flow | Zell Liew:
   https://zellwk.com/blog/git-flow/
.. _Using git-flow to automate your git branching workflow:
   https://jeffkreeftmeijer.com/git-flow/

If you want to contribute to ADOpy, you can start with forking the ADOpy
repository in the GitHub. Before you start writing a code, please make a new
issue for what you want to do. Then, following the Gitflow, you should make a
proper branch to work with. We only allow feature branches for who is not
authorized. If your job is done, you can make a pull request on the ADOpy
repository and let us review it before merging it.

