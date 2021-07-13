---
title: 'A Fast Iterative Method Python package'
tags:
  - Python
  - eikonal
  - partial differential equations
  - cuda
authors:
  - name: Thomas Grandits
    affiliation: 1
affiliations:
 - name: Institute of Computer Graphics and Vision, TU Graz
   index: 1
date: July 2021
bibliography: paper.bib
---

# Summary


The anisotropic eikonal equation is a non-linear partial differential equation, given by
\begin{equation*}
   \left\{
   \begin{array}{rll}
   \left<\nabla \phi, D \nabla \phi \right> &= 1 \quad &\text{on} \; \Omega \\
   \phi(\mathbf{x}_0) &= g(\mathbf{x}_0) \quad &\text{on} \; \Gamma \subset \Omega
   \end{array}
   \right. .
\end{equation*}
In practice, this problem is often associated with computing the earliest arrival times $\phi$ of a wave from a set of given starting points $\mathbf{x}_0$ through a heterogeneous medium (i.e. different velocities are assigned throughout the medium). 
This equation yields infinitely many weak solutions [@evans_1997_partial] and can thus not be straight-forwardly solved using standard Finite Element approaches.

``fim-python`` implements the Fast Iterative Method (FIM) purely in Python to solve the anisotropic eikonal equation by finding its unique viscosity solution.
In this scenario, we compute $\phi$ on tetrahedral/triangular meshes or line networks for a given $D$, $\mathbf{x}_0$ and $g$.
The method is implemented both on the CPU using [``numba``](https://numba.pydata.org/) and [``numpy``](https://numpy.org/), as well as the GPU with the help of [``cupy``](https://cupy.dev/) (depends on [CUDA](https://developer.nvidia.com/cuda-toolkit)).
The library is meant to be easily and rapidly used for repeated evaluations on a mesh.

The eikonal equation has many practical applications, including cardiac electrophysiology, image processing and geoscience, to approximate wave propagation through a medium.
In the example of cardiac electrophysiology [@franzone2014mathematical], the electrical activation times $\phi$ are computed throughout the anisotropic heart muscle with varying conduction velocities $D$.

The FIM locally computes an update rule to find the path the wavefront will take through a single element.
Since the algorithm is restricted to linear elements, the path through an element will also be a straight line.
In the case of tetrahedral domains, the FIM thus tries to find the path of the linear update from a face spanned by three vertices $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ to the opposite vertex $\mathbf{v}_4$.
\autoref{fig:update} visualizes the update.
For triangles and lines, the algorithm behaves similarly but the update origin is limited to a side or vertex respectively.
The exact equations used to solve this problem in this repository were previously described (among others) in [@grandits_inverse_2020].

![Update inside a single tetrahedron\label{fig:update}](docs/figs/update_fig.jpg "Update inside a single tetrahedron"){ width=33% }


Two different methods are implemented in ``fim-python``:
In the *Jacobi* method, the above local update rule is computed for all elements in each iteration until the change between two subsequent iterations is smaller than a chosen $\varepsilon$.
This version of the algorithm is bested suited for the GPU, since it is optimal for a SIMD (single instruction multiple data) architecture.
The *active list* method is more closely related to the method presented in [@fu_fast_2013]:
We keep track of all vertices that require a recomputation in the current iteration on a so-called active list which we keep up-to-date. 

# Statement of need

The publicly available libraries to solve the eikonal equation have at least one of the following restrictions:

* Isotropic eikonal equation only ($D = c I$ for $c \in \mathbb{R}$ and $I$ being the identity matrix), solved using Fast Marching [@sethian_fast_1996]
* Limited support for multiple source points
* 2D only
* Restricted to uniform or structured grids
* No Python implementation
* CPU only

``fim-python`` tries to address all these issues and makes installing straight-forward by also providing the package over [PyPI](https://pypi.org/), which can be installed using pip:

```{bash}
pip install cython
pip install fim-python[gpu]
```

# References
