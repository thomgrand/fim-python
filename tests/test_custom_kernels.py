#import unittest
import pytest
#from .
import numpy as np
import os
from fimpy.fim_cutils import compute_perm_mask, compute_point_elem_map_c, compute_neighborhood_map_c
from scipy.spatial import Delaunay
from itertools import combinations

try:
    import cupy as cp
    from fimpy.cupy_kernels import compute_perm_kernel_str, compute_perm_kernel_shared
    cupy_enabled = True
    #raise ImportError("Test") #For testing the CPU only version
except ImportError as err:
    print("Cupy import failed. The tests will skip the cupy tests")
    cupy_enabled = False

class TestCustomKernels():

    def create_mesh(self, elem_dims, resolution):
        x = np.linspace(-1, 1, num=resolution)
        pts = np.stack(np.meshgrid(*((x,) * elem_dims), indexing='ij'), axis=-1).reshape([-1, elem_dims])
        elems = Delaunay(pts).simplices
        inds = np.arange(elem_dims+1)
        assert(elem_dims in [2, 3])

        if elem_dims == 2:
            perms = np.array([[0, 1, 2],
                              [0, 2, 1],
                              [2, 1, 0]])
        else:
            perms = np.array([[0, 1, 2, 3],
                              [0, 1, 3, 2],
                              [0, 3, 2, 1],
                              [3, 1, 2, 0]])
        elems_perm = elems[np.arange(elems.shape[0])[:, np.newaxis, np.newaxis], perms[np.newaxis]]

        return pts, elems, elems_perm

    def comp_point_elem_map(self, elems, nr_points):
        #TODO: Export into function?
        max_point_elem_ratio = np.max(np.unique(elems, return_counts=True)[1])
        point_elem_map = np.zeros([nr_points, max_point_elem_ratio], dtype=np.int32)
        compute_point_elem_map_c(elems, point_elem_map)
        return point_elem_map

    def comp_neighborhood_map(self, elems, nr_points, elem_dims):
        max_point_elem_ratio = np.max(np.unique(elems, return_counts=True)[1])
        nh_map = np.zeros(shape=[nr_points, max_point_elem_ratio * elem_dims], dtype=np.int32)

        nh_map = np.array(compute_neighborhood_map_c(elems, nh_map))

        nh_map = np.sort(nh_map, axis=-1)

        # There may be cases where the ratio was an overestimate
        while nh_map.shape[1] > 1 and np.all(nh_map[..., -1] == nh_map[..., -2]):
            nh_map = nh_map[..., :-1]

        return nh_map

    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
    @pytest.mark.parametrize('elem_dims', [3, 4])
    @pytest.mark.parametrize('resolution', [5, 10, 15])
    @pytest.mark.parametrize('nr_active_inds', [1, 20, 100])
    @pytest.mark.parametrize('parallel_blocks', [1, 4, 16])
    @pytest.mark.parametrize('threads_x', [1, 16, 32])
    @pytest.mark.parametrize('threads_y', [1]) #, 16, 32]) #New kernel with binary search does not use this
    @pytest.mark.parametrize('shared_buf_size', [1, 128, 2048])
    def test_perm_kernel_gpu2(self, elem_dims, resolution, nr_active_inds, parallel_blocks, threads_x, threads_y, shared_buf_size):
        pts, elems, elems_perm = self.create_mesh(elem_dims-1, resolution)
        nr_pts = pts.shape[0]

        elems_perm = cp.array(elems_perm)
        point_elem_map = cp.array(self.comp_point_elem_map(elems, nr_pts))
        active_inds = cp.sort(cp.array(np.random.choice(nr_pts, np.minimum(nr_active_inds, nr_pts), replace=False)))
        active_elem_inds = cp.unique(point_elem_map[active_inds].reshape([-1]))
        active_elems_perm = elems_perm[active_elem_inds]

        perm_kernel = cp.RawKernel(compute_perm_kernel_shared.replace("{active_perms}", str(elem_dims)).replace("{parallel_blocks}", str(parallel_blocks)).replace("{shared_buf_size}", str(shared_buf_size)), 
                                    'perm_kernel', options=('-std=c++11',), backend='nvcc')
        perm_kernel.compile()
        perm_mask_output = cp.zeros(shape=active_elems_perm.shape[0:2], dtype=bool)
        perm_kernel((parallel_blocks,), (threads_x,threads_y, 1), (cp.ascontiguousarray(active_elems_perm[..., -1]), active_inds.astype(cp.int32), perm_mask_output, active_elems_perm.shape[0], active_inds.size))
        #Broadcasted version is the ground truth
        perm_mask_gt = np.any(active_elems_perm[..., -1, np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1)
        assert(np.all(perm_mask_output == perm_mask_gt))

if __name__ == "__main__":
    self = TestCustomKernels()
    parallel_blocks = 4
    threads_x = 32
    threads_y = 1
    shared_buf_size = 1024
    nr_active_inds = 5
    elem_dims = 3
    resolution = 15
    pts, elems, elems_perm = self.create_mesh(elem_dims-1, resolution)
    nr_pts = pts.shape[0]

    elems_perm = cp.array(elems_perm)
    point_elem_map = cp.array(self.comp_point_elem_map(elems, nr_pts))
    active_inds = cp.sort(cp.array(np.random.choice(nr_pts, np.minimum(nr_active_inds, nr_pts), replace=False)))
    active_elem_inds = cp.unique(point_elem_map[active_inds].reshape([-1]))
    active_elems_perm = elems_perm[active_elem_inds]

    perm_kernel = cp.RawKernel(compute_perm_kernel_shared.replace("{active_perms}", str(elem_dims)).replace("{parallel_blocks}", str(parallel_blocks)).replace("{shared_buf_size}", str(shared_buf_size)), 
                                'perm_kernel')
    perm_kernel.compile()
    perm_mask_output = cp.zeros(shape=active_elems_perm.shape[0:2], dtype=bool)
    perm_kernel((parallel_blocks,), (threads_x,threads_y, 1), (cp.ascontiguousarray(active_elems_perm[..., -1]), active_inds.astype(cp.int32), perm_mask_output, active_elems_perm.shape[0], active_inds.size))
    #Broadcasted version is the ground truth
    perm_mask_gt = np.any(active_elems_perm[..., -1, np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1)
    assert(np.all(perm_mask_output == perm_mask_gt))