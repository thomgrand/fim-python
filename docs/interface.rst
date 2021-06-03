
Interface Methods
=================
All different solvers can be generated using the interface class.
Note that if you specify the gpu interface, but your system does not support it (or you did not install it), you will only get a cpu solver.

.. automethod:: fimpy.solver.FIMPY.create_fim_solver

Computing the anisotropic eikonal equation can be easily achieved by calling :meth:`fimpy.fim_base.FIMBase.comp_fim` on the returned solver.

.. automethod:: fimpy.fim_base.FIMBase.comp_fim

.. toctree::
   :maxdepth: 2
   :caption: Contents:
