Detailed Description
====================

The Fast Iterative Method locally computes an update rule, rooted in the Hamilton-Jacobi formalism of the eikonal problem, computing the path the front-wave will take through the current element.
Since the algorithm is restricted to linear Lagrangian :math:`\mathcal{P}^1` elements, the path through an element will also be a line.
To demonstrate the algorithm, consider a tetrahedron spanned by the four corners :math:`\mathbf{v}_1` through :math:`\mathbf{v}_4`.
For the earliest arrival times associated to each corner, we will use the notation :math:`\phi_i = \phi(\mathbf{v}_i)`.
The origin of a linear update from a face spanned by three vertices :math:`\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3` to the fourth :math:`\mathbf{v}_4` is required to be inside said face.
Mathematically this is described by the following set:

.. math::
    \Delta_k = \left\{ \left( \lambda_1, \ldots, \lambda_k \right)^\top \middle\vert \sum_{i=1}^k \lambda_i = 1 \land \lambda_i \ge 0 \right\}

The earliest arrival time :math:`\phi_4` can be found by solving the minimization problem which constitutes the local update rule

.. math::
    \phi_4 = \min_{\lambda_1, \lambda_2} \, \sum_{i=1}^3\lambda_i \phi_i + \sqrt{\mathbf{e}_{\Delta}^\top D^{-1} \mathbf{e}_{\Delta}} \quad \text{s.t.: } \, \left( \lambda_1, \lambda_2, \lambda_3 \right)^\top \in \Delta_3

for :math:`\lambda_3 = 1 - \lambda_1 - \lambda_2` and :math:`\mathbf{e}_{\Delta} = \mathbf{v}_4 - \sum_{i=1}^3 \lambda_i \mathbf{v}_i`.
The picture below visualizes the update.

.. image:: figs/update_fig.jpg
    :width: 300
    :alt: Update inside a single tetrahedron
    :align: center

When updating a tetrahedron, we compute the update of each of the faces to the opposite vertex.
The newly calculated value :math:`\phi_4` will only become the new value if it is strictly smaller than the old value.

For triangles and lines, the algorithm behaves similarly but the update origin is limited to a side or vertex respectively.
The internally implemented updates in the algorithm to solve the minimization problem are similar to the ones reported in `An inverse Eikonal method for identifying ventricular activation sequences from epicardial activation maps <https://www.sciencedirect.com/science/article/pii/S0021999120304745>`_.


Jacobi vs. Active List Method
-----------------------------
Two different methods are implemented in the repository:
In the *Jacobi* method, the above local update rule is computed for all elements in each iteration until the change between two subsequent iterations is smaller than ``convergence_eps`` (:math:`10^{-9}` by default).
This version of the algorithm is bested suited for the GPU, since it is optimal for a SIMD (single instruction multiple data) architecture.

The *active list* method is more closely related to the method presented in the `paper <https://epubs.siam.org/doi/abs/10.1137/120881956>`_:
We keep track of all vertices that will be updated in the current iteration. 
Initially, we start off with the neighbor nodes to the initial points :math:`\mathbf{x}_0`.
Once convergence has been reached for a vertex on the active list (according to ``convergence_eps``), its neighboring nodes will be recomputed and if the new value is smaller than the old, they will be added onto the active list.
Convergence is achieved once the active list is empty.

The active list method computes much fewer updates, but has the additional overhead of keeping track of its active list, ill-suited for the GPU.
For larger meshes, the active list is still a better choice, but comes at the additional cost of a setup time (see :doc:`Benchmark <benchmark>`), making it best suited for repeated queries of the same mesh with different :math:`D, g, \mathbf{x}_0`.
