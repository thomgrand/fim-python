Benchmark
============

Both the *Jacobi* and *active list* methods have been tested on heterogeneous ND cubes.
Here you can see the comparison of both their run- and setup-time.
The dashed lines show the performance of the implementation using active lists, the solid lines use the Jacobi method (see :doc:`Detailed Description <detailed_description>` for more info).

Runtime
--------

Below you can see a performance benchmark of the library for tetrahedral domains (cube in ND), triangular surfaces (plane in ND), and line networks (randomly sampled point cloud in the ND cube with successive minimum spanning tree) from left to right.
In all cases, :math:`\mathbf{x}_0` was placed in the middle of the domain.

.. image:: figs/benchmark_gpu.jpg
    :alt: Benchmark GPU
    :align: center

.. image:: figs/benchmark_cpu.jpg
    :alt: Benchmark CPU
    :align: center

The library works for an arbitrary number of dimensions (manifolds in N-D), but the versions for 2 and 3D received a few optimized kernels that speed up the computations.

Setup Time
----------

The active list method additionally needs to create a few mesh specific fields before computation to efficiently update the active list.
This makes it best suited for repeated queries of the same mesh with different :math:`D, g, \mathbf{x}_0`.
The figure below shows the setup time for both methods.

.. image:: figs/benchmark_gpu_setup.jpg
    :alt: Setup Time GPU
    :align: center