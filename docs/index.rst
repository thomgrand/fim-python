.. FIM Python documentation master file, created by
   sphinx-quickstart on Mon May 17 21:13:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FIM-Python (fimpy) documentation!
======================================

.. contents:: Quick Start
    :depth: 3

Introduction
------------

This library implements the Fast Iterative method that solves the anisotropic eikonal equation on 
`triangulated surfaces <https://epubs.siam.org/doi/abs/10.1137/100788951>`_, 
`tetrahedral meshes <https://epubs.siam.org/doi/abs/10.1137/120881956>`_ and line networks 
(equivalent to the `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_ in this case),
for arbitrary dimensions.

The anisotropic eikonal equation that is solved, is given by the partial differential equation

.. math::
   \left\{
   \begin{array}{rll}
   \left<\nabla \phi, D \nabla \phi \right> &= 1 \quad &\text{on} \; \Omega \\
   \phi(\mathbf{x}_0) &= g(\mathbf{x}_0) \quad &\text{on} \; \Gamma
   \end{array}
   \right. .

The library computes :math:`\phi` for a given :math:`D`, :math:`\mathbf{x}_0` and :math:`g`.

   

Usage
---------------------
The following shorthand notations are important to know:

- :math:`n`: Number of points
- :math:`m`: Number of elements
- :math:`d`: Dimensionality of the points and metrics (:math:`D \in \mathbb{R}^{d \times d}`)
- :math:`d_e`: Number of vertices per elements (2, 3 and 4 for lines, triangles and tetrahedra respectively)
- :math:`k`: Number of discrete points :math:`\mathbf{x}_0 \in \Gamma` and respective values in :math:`g(\mathbf{x}_0)`
- :math:`M := D^{-1}`: The actual metric used in all computations. This is is computed internally and automatically by the library (no need for you to invert :math:`D`)
- precision: The chosen precision for the solver at the initialization

This example computes the solution to the anisotropic eikonal equation for a simple square domain
:math:`\Omega = [-1, 1]^2`, with :math:`n = 50^2, d = 2, d_e = 3` and a given isotropic :math:`D`. 
This example requires additionally matplotlib and scipy.


.. include:: example.inc

You should see the following figure with the computed :math:`\phi` for the given :math:`D = c I`.


.. image:: figs/usage_example.jpg
  :alt: Usage example

.. include:: installation.rst






.. toctree::
   :maxdepth: 2
   :caption: Detailed Contents

   interface.rst
   detailed_description.rst


Module API
--------------

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :caption: Module

   fimpy


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
