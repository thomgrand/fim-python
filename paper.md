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

The Fast Iterative Method (FIM) [@fu_fast_2013], [@fu_fast_2011] is an efficient algorithm to solve the anisotropic eikonal equation, given by the partial differential equation
\begin{equation*}
   \left\{
   \begin{array}{rll}
   \left<\nabla \phi, D \nabla \phi \right> &= 1 \quad &\text{on} \; \Omega \\
   \phi(\mathbf{x}_0) &= g(\mathbf{x}_0) \quad &\text{on} \; \Gamma
   \end{array}
   \right. .
\end{equation*}

The FIM locally computes an update rule, rooted in the Hamilton-Jacobi formalism of the eikonal problem, computing the path the front-wave will take through the current element.
Since the algorithm is restricted to linear Lagrangian $\mathcal{P}^1$ elements, the path through an element will also be a straight line.
To demonstrate the algorithm for tetrahedral domains, consider a single tetrahedron spanned by the four corners $\mathbf{v}_1$ through $\mathbf{v}_4$.
The FIM then tries to find the the origin of the linear update from a face spanned by three vertices $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ to the opposite vertex $\mathbf{v}_4$.
\autoref{fig:update} visualizes the update.
For triangles and lines, the algorithm behaves similarly but the update origin is limited to a side or vertex respectively.

![Update inside a single tetrahedron\label{fig:update}](docs/figs/update_fig.jpg "Update inside a single tetrahedron"){ width=40% }

The eikonal equation has many practical applications, including cardiac electrophysiology, image processing and geoscience, to approximate wave propagation through a medium.
In practice, this problem is often associated with computing the earliest arrival times $\phi$ of a wave from a set of given starting points $\mathbf{x}_0$ through a heterogeneous medium (i.e. different velocities are assigned throughout the medium). 
In the example of cardiac electrophysiology [@franzone2014mathematical], the electrical activation times $\phi$ are computed throughout the anisotropic heart muscle with varying conduction velocities $D$.

``fim-python`` implements the FIM to compute $\phi$ on triangulated/tetrahedral meshes or line networks for a given $D$, $\mathbf{x}_0$ and $g$.
The method is implemented both on the CPU using [``numba``](https://numba.pydata.org/) and [``numpy``](https://numpy.org/), as well as the GPU with the help of [``cupy``](https://cupy.dev/).
The library is meant to be easily and rapidly used for repeated evaluations on a mesh.


Two different methods are implemented in ``fim-python``:
In the *Jacobi* method, the above local update rule is computed for all elements in each iteration until the change between two subsequent iterations is smaller than $\varepsilon$.
This version of the algorithm is bested suited for the GPU, since it is optimal for a SIMD (single instruction multiple data) architecture.
The *active list* method is more closely related to the method presented in [@fu_fast_2013]:
We keep track of all vertices that require a recomputation in the current iteration on a so-called active list which we keep up-to-date. 

# Statement of need

The publicly available libraries for this problem, have several restrictions:

* Isotropic eikonal equation only ($D = c I$ for $c \in \mathbb{R}$ and $I$ being the identity matrix), solved using Fast Marching [@sethian_fast_1996]
* Single source points ($|\mathbf{x}_0| = 1$)
* 2D only
* Restricted to uniform or structured grids
* No Python implementation
* CPU only

``fim-python`` tries to address all these issues and makes installing straight-forward by also providing the package over [PyPI](https://pypi.org/), which can be installed using pip:

```{bash}
pip install cython
pip install fimpy[gpu]
```

# References
