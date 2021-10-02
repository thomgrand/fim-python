Installation
--------------

To install, either clone the repository and install it:

.. code-block:: bash

    git clone https://github.com/thomgrand/fim-python .
    pip install -e .[gpu]


or simply install the library over `PyPI <https://pypi.org>`_.

.. code-block:: bash

    pip install fim-python[gpu]

.. note:: 

    Installing the GPU version might take a while since many ``cupy`` modules are compiled using your system's ``nvcc`` compiler.
    You can install the ``cupy`` binaries first as mentioned `here <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_, before installing ``fimpy``.