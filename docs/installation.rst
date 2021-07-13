Installation
--------------

The only prerequisite before installing is cython to compile some of the files

.. code-block:: bash

    pip install cython

All remaining dependencies should be installed directly using pip.
To install, either clone the repository and install it:

.. code-block:: bash

    git clone https://github.com/thomgrand/fim-python .
    pip install -e .[gpu]


or simply via pip 

.. code-block:: bash

    pip install fim-python[gpu]

.. note:: 

    Installing the GPU version might take a while since many ``cupy`` modules are compiled using your system's ``nvcc`` compiler.
    You can install the ``cupy`` binaries first as mentioned `here <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_, before installing ``fimpy``.