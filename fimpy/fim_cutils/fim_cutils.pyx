# cython: infer_types=True
# distutils: language=c++
import numpy as np
cimport cython
from cython.parallel import prange
from libcpp cimport bool as bool_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_point_elem_map_c(int[:, :] elems, int[:, :] point_elem_map):
    """Computes the point to element map, i.e. for each point the elements it is contained in.
    This function assumes all points are contained in at least one element.

    Parameters
    ----------
    elems : ndarray (int)
        An [M, d_e] array holding the mesh connectivity
    point_elem_map : ndarray (int)
        An [N, ?] array that will hold the result. 
        Note that the last shape has to be at least long enough to hold the map for the point with the maximum number of contained elements.
        All points that are not contained in that many elements, will fill the array with the last occuring element index.
    """
    nr_elems = elems.shape[0]
    nr_points = point_elem_map.shape[0]
    max_point_elem_ratio = point_elem_map.shape[1]
    current_offsets_arr = np.zeros_like(point_elem_map[..., 0])

    cdef int[::1] current_offsets = current_offsets_arr
    cdef int offset, point
    for elem_i in range(nr_elems):
        for elem_j in range(elems.shape[1]):
            point = elems[elem_i, elem_j]
            offset = current_offsets[point]
            point_elem_map[point, offset] = elem_i
            current_offsets[point] += 1

    point_elem_map = point_elem_map[:, :np.max(current_offsets)]
    for point_i in range(nr_points):
        point_elem_map[point_i, current_offsets[point_i]:] = point_elem_map[point_i, current_offsets[point_i]-1]

    return point_elem_map

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_neighborhood_map_c(int[:, ::1] elems, int[:, ::1] nh_map):
    """Computes the point to neighborhood map, i.e. for each point the indices of its neighboring points.
    This function assumes all points are contained in at least one element.

    Parameters
    ----------
    elems : ndarray (int)
        An [M, d_e] array holding the mesh connectivity
    nh_map : ndarray (int)
        An [N, ?] array that will hold the result. 
        Note that the last shape has to be at least long enough to hold the map for the point with the maximum number of neighbors.
        All points that do not have that many neighbors, will fill the array with the last occuring element index.
    """
    nh_map[:] = -1
    nr_points = nh_map.shape[0]
    nr_elems = elems.shape[0]
    current_offsets_arr = np.zeros_like(nh_map[..., 0])
    elem_range = np.arange(elems.shape[1], dtype=np.int32)
    cdef int[::1] inds, points, current_offsets
    cdef int point, offset, elem_i, single_ind, i, inds_size, elem_j, elem_dims
    elem_dims = elems.shape[1]
    current_offsets = current_offsets_arr
    inds = elem_range
    inds_size = elem_range.size
    for elem_i in range(nr_elems):
        points = elems[elem_i]
        for elem_j in prange(elem_dims, nogil=True):
            point = points[elem_j]
            #inds_arr = np.delete(elem_range, elem_j)
            #inds = inds_arr
            offset = current_offsets[point]
            #inds_size = inds.size
            for i in range(inds_size):
                if i != elem_j:
                    single_ind = inds[i]
                    nh_map[point, offset+i] = elems[elem_i, single_ind]
            current_offsets[point] += inds_size
    
    cdef int point_i, neigh_i, unique_neighs_size
    cdef int[::1] unique_neighs_view
    for point_i in range(nr_points):
        unique_neighs = np.unique(nh_map[point_i])
        unique_neighs = unique_neighs[unique_neighs != -1]
        unique_neighs_size = unique_neighs.size
        unique_neighs_view = unique_neighs
        for neigh_i in prange(nh_map.shape[1], nogil=True):
            if neigh_i < unique_neighs_size:
                nh_map[point_i, neigh_i] = unique_neighs_view[neigh_i]
            else:
                nh_map[point_i, neigh_i] = unique_neighs_view[unique_neighs_size-1]


    return nh_map

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_perm_mask(int[:, ::1] active_elems_perm, int[::1] active_inds):
    """Returns a mask with all active_elems_perm that have at least one active_inds in them. 
    Equivalent to np.any(active_elems[..., np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1), but does not require the full memory.

    Parameters
    ----------
    active_elems_perm : ndarray (int)
        An [a, b] array holding all the permuted element connectivities
    active_inds : ndarray (int)
        An [c] array with all active indices (points).

    Returns
    -------
    np.ndarray (bool)
        [a, b] array holding the computed mask.
    """
    cdef int nr_elems, nr_perms, nr_active_inds, elem_i, perm_i, active_i, active_ind
    nr_elems = active_elems_perm.shape[0]
    nr_perms = active_elems_perm.shape[1]
    nr_active_inds = active_inds.size
    perm_mask_arr = np.zeros(shape=[nr_elems, nr_perms], dtype=bool)
    cdef bool_t[:, ::1] perm_mask = perm_mask_arr

    for elem_i in prange(nr_elems, nogil=True):
        for perm_i in range(nr_perms):
            for active_i in range(nr_active_inds):
                if active_inds[active_i] == active_elems_perm[elem_i, perm_i]:
                    perm_mask[elem_i, perm_i] = True
                    break

    return perm_mask_arr
