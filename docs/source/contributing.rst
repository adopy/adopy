Contributing
============

You can set up a developmental environment using pipenv.

.. code-block:: bash

   # Clone the repository from Github.
   git clone https://github.com/JaeyeongYang/adopy.git

   # Set the working directory to the cloned repository.
   cd adopy

   # Install dev dependencies with pipenv
   pipenv install --dev

   # Install adopy with flit with symlink
   pipenv run flit install -e
