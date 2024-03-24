# Fast Iterative Method - Numpy/Cupy
This repository implements the Fast Iterative Method on [tetrahedral domains](https://epubs.siam.org/doi/abs/10.1137/120881956) and [triangulated surfaces](https://epubs.siam.org/doi/abs/10.1137/100788951) purely in python both for CPU (numpy) and GPU (cupy). The main focus is however on the GPU implementation, since it can be better exploited for very large domains.

[![codecov](https://codecov.io/gh/thomgrand/fim-python/branch/master/graph/badge.svg?token=DG05WR5030)](https://codecov.io/gh/thomgrand/fim-python)
[![CI Tests](https://github.com/thomgrand/fim-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/thomgrand/fim-python/actions/workflows/python-package.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03641/status.svg)](https://doi.org/10.21105/joss.03641)

# Details
The anisotropic eikonal equation is given by

![$$\left<D \nabla \phi, \nabla \phi\right> = 1$$](https://latex.codecogs.com/svg.latex?\Large&space;\left%3CD%20\nabla%20\phi,%20\nabla%20\phi\right%3E%20=%201)


for given boundary conditions 

![$$\phi(\mathbf{x}_0) = g(\mathbf{x}_0)$$](https://latex.codecogs.com/svg.latex?\Large\phi(\mathbf{x}_0)%20=%20g(\mathbf{x}_0))

For a given anisotropic velocity, this can calculate the geodesic distance between a set of ![$\mathbf{x}_0$](https://latex.codecogs.com/svg.latex?\Large\mathbf{x}_0) and all points on the domain like shown in the figure.

![Preview Image](docs/figs/usage_example.jpg)

Note that when using multiple ![$\mathbf{x}_0$](https://latex.codecogs.com/svg.latex?\Large\mathbf{x}_0), they are not guaranteed to be in the final solution if they are not a valid viscosity solution. A recommended read for more details on the subject is:  
Evans, Lawrence C. "Partial differential equations." *Graduate studies in mathematics* 19.2 (1998).

# Installation

The easiest way to install the library is using pip
```bash
pip install fim-python[gpu] #GPU version
```

If you don't have a compatible CUDA GPU, you can install the CPU only version to test the library, but the performance won't be comparable to the GPU version (see [Benchmark](#benchmark)).

```bash
pip install fim-python #CPU version
```

# Usage

The main interface to create a solver object to use is [`create_fim_solver`](https://fim-python.readthedocs.io/en/latest/interface.html#fimpy.solver.create_fim_solver)

```python
from fimpy.solver import create_fim_solver

#Create a FIM solver, by default the GPU solver will be called with the active list
#Set device='cpu' to run on cpu and use_active_list=False to use Jacobi method
fim = create_fim_solver(points, elems, D)
```

Example
-------

The following code reproduces the [above example](#details)

```python
import numpy as np
import cupy as cp
from fimpy.solver import create_fim_solver
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

#Create triangulated points in 2D
x = np.linspace(-1, 1, num=50)
X, Y = np.meshgrid(x, x)
points = np.stack([X, Y], axis=-1).reshape([-1, 2]).astype(np.float32)
elems = Delaunay(points).simplices
elem_centers = np.mean(points[elems], axis=1)

#The domain will have a small spot where movement will be slow
velocity_f = lambda x: (1 / (1 + np.exp(3.5 - 25*np.linalg.norm(x - np.array([[0.33, 0.33]]), axis=-1)**2)))
velocity_p = velocity_f(points) #For plotting
velocity_e = velocity_f(elem_centers) #For computing
D = np.eye(2, dtype=np.float32)[np.newaxis] * velocity_e[..., np.newaxis, np.newaxis] #Isotropic propagation

x0 = np.array([np.argmin(np.linalg.norm(points, axis=-1), axis=0)])
x0_vals = np.array([0.])

#Create a FIM solver, by default the GPU solver will be called with the active list
fim = create_fim_solver(points, elems, D)
phi = fim.comp_fim(x0, x0_vals)

#Plot the data of all points to the given x0 at the center of the domain
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
cont_f1 = axes[0].contourf(X, Y, phi.get().reshape(X.shape))
axes[0].set_title("Distance from center")

cont_f2 = axes[1].contourf(X, Y, velocity_p.reshape(X.shape))
axes[1].set_title("Assumed isotropic velocity")
plt.show()
```

A general rule of thumb: If you only need to evaluate the eikonal equation once for a mesh, the Jacobi version (`use_active_list=False`) will probably be quicker since its initial overhead is low.
Repeated evaluations with different ![$\mathbf{x}_0$](https://latex.codecogs.com/svg.latex?\Large\mathbf{x}_0) or ![$D$](https://latex.codecogs.com/svg.latex?\Large%20D) favor the active list method for larger meshes.  
On the CPU, `use_active_list=True` outperforms the Jacobi approach for almost all cases.

# Documentation

[https://fim-python.readthedocs.io/en/latest](https://fim-python.readthedocs.io/en/latest)

# Citation

If you find this work useful in your research, please consider citing the [paper](https://doi.org/10.21105/joss.03641) in the [Journal of Open Source Software](https://joss.theoj.org/)
```bibtex
@article{grandits_fast_2021,
  doi = {10.21105/joss.03641},
  url = {https://doi.org/10.21105/joss.03641},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {66},
  pages = {3641},
  author = {Thomas Grandits},
  title = {A Fast Iterative Method Python package},
  journal = {Journal of Open Source Software}
}
```

# Benchmark

Below you can see a performance benchmark of the library for tetrahedral domains (cube in ND), triangular surfaces (plane in ND), and line networks (randomly sampled point cloud in the ND cube with successive minimum spanning tree) from left to right.
In all cases, ![$\mathbf{x}_0$](https://latex.codecogs.com/svg.latex?\Large\mathbf{x}_0) was placed in the middle of the domain.
The dashed lines show the performance of the implementation using active lists, the solid lines use the Jacobi method (computing all updates in each iteration).

![Preview](docs/figs/benchmark_gpu.jpg)

![Preview](docs/figs/benchmark_cpu.jpg)

The library works for an arbitrary number of dimensions (manifolds in N-D), but the versions for 2 and 3D received a few optimized kernels that speed up the computations.

The steps to reproduce the benchmarks can be found in the documentation at [https://fim-python.readthedocs.io/en/latest/benchmark.html](https://fim-python.readthedocs.io/en/latest/benchmark.html)

# Contributing

See [Contributing](CONTRIBUTING.md) for more information on how to contribute.

# License

This library is licensed under the [GNU Affero General Public License](LICENSE). 
If you need the library issued under another license for commercial use, you can contact me via e-mail [tomdev (at) gmx.net](mailto:tomdev@gmx.net).
