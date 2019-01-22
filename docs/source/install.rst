Installation
============

Using pip
---------

Using pip, you can install adopy from PyPI.

.. code:: bash

   pip install adopy

Instead, you can install it from source in the GitHub repository.

.. code-block:: bash

    # Clone the repository from Github.
    $ git clone https://github.com/adopy/adopy.git

    # Set the working directory to the cloned repository.
    $ cd adopy

    # Install ADOpy with pip
    $ pip install .

Using conda
-----------

Using `Anaconda`_ makes it possible to use its optimized packages in `Anaconda
Cloud`_ and
convinient virtual environment managements.
To use |conda|_, an open source package management system and environment
management system, you should have installed `Anaconda`_.

.. _Anaconda: https://docs.continuum.io/anaconda/
.. _Anaconda Cloud: https://docs.continuum.io/anaconda-cloud/
.. |conda| replace:: ``conda``
.. _conda: https://conda.io/en/latest/

.. code:: bash

   # Make a separate virtual environment for ADOpy (optional)
   conda create -n adopy
   source activate adopy

   # Install dependencies from Anaconda Cloud
   conda install numpy scipy pandas

   # Install ADOpy from PyPI
   pip install adopy

For developers
--------------

To unify the developmental environment, we use |pipenv|_. The default Python
version is set to Python 3.5, which is the minimal requirement for ADOpy.

.. |pipenv| replace:: ``pipenv``
.. _pipenv: https://pipenv.readthedocs.io/en/latest/

.. code:: bash

   # Prepare a virtual environment for development
   pipenv install --dev

   # Activate the environment
   pipenv shell
