"""This subpackage contains the Cython implementations for multiple functions: 
    * Generating the point to element maps
    * Generating the point to neighborhood maps
    * Computation of the permutation/active indices mask

    All of these are only used in solvers using the active list.
"""

from .fim_cutils import compute_point_elem_map_c, compute_neighborhood_map_c, compute_perm_mask
