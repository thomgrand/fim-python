#import unittest
import pytest
#from .
from fimpy.solver import FIMPY
import numpy as np
import os
import scipy.io as sio

try:
    import cupy as cp
    cupy_enabled = True
    #raise ImportError("Test") #For testing the CPU only version
except ImportError as err:
    print("Cupy import failed. The tests will skip the cupy tests")
    cupy_enabled = False

class TestFIMSolversInit():
    
    @pytest.mark.parametrize('init_D', [True, False])
    @pytest.mark.parametrize('dims', [1, 2, 3])
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('use_active_list', [True, False])
    def test_init_cpu(self, dims, init_D, precision, use_active_list):
        points = np.tile(np.linspace(0, 1, num=4)[:(dims+1)][:, np.newaxis], [1, dims])
        elems = np.arange(points.shape[0])[np.newaxis]
        D = None
        if init_D:
            D = np.eye(dims)[np.newaxis]

        fim_solver = FIMPY.create_fim_solver(points, elems, D, device='cpu', precision=precision, use_active_list=use_active_list)

    @pytest.mark.parametrize('init_D', [True, False])
    @pytest.mark.parametrize('dims', [1, 2, 3])
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('use_active_list', [True, False])
    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
    def test_init_gpu(self, dims, init_D, precision, use_active_list):
        points = np.tile(np.linspace(0, 1, num=4)[:(dims+1)][:, np.newaxis], [1, dims])
        elems = np.arange(points.shape[0])[np.newaxis]
        D = None
        if init_D:
            D = np.eye(dims)[np.newaxis]

        fim_solver = FIMPY.create_fim_solver(points, elems, D, device='gpu', precision=precision, use_active_list=use_active_list)

    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    def test_error_init(self, precision, device='cpu'):
        points = np.array([0.])
        elems = np.array([0])

        #Wrong dimensions
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems)

        #Points not numeric
        points = np.array([[0, 0], [1, 0]]).astype(np.int32)
        elems = np.array([[0, 1]])
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems)

        #D and elems not matching
        points = np.array([[0., 0], [1, 0]])
        elems = np.array([[0, 1]])
        D = np.tile(np.eye(2)[np.newaxis], [2, 1])
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems, D)

        #elems references non-existant points
        points = np.array([[0., 0.], [1., 0.]])
        elems = np.array([[0, 1, 2]])
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems, precision=precision, device=device)

        
        #points not contained in any element
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        elems = np.array([[0, 1]])
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems, precision=precision, device=device)

        #Unsupported element dimensions (Polygons and other elements)
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
        elems = np.array([[0, 1, 2, 3, 4]])
        with pytest.raises(Exception):
            FIMPY.create_fim_solver(points, elems, precision=precision, device=device)
        

    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    def test_error_init_gpu(self, precision):
        self.test_error_init(precision, 'gpu')

from generate_test_data import test_dims, test_elem_dims, test_resolutions, elem_fnames

class TestFIMSolversComputations():

    test_dir = os.path.join(__file__, "data")

    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('dims', test_dims)
    @pytest.mark.parametrize('elem_dims', test_elem_dims)
    @pytest.mark.parametrize('use_active_list', [True, False])
    def test_comp(self, dims, elem_dims, precision, use_active_list, device='cpu'):
        if (dims < elem_dims - 1):
            return

        np.random.seed(0)
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        for resolution in test_resolutions[elem_dims-1]:
            fname = "elem_dims_%d_dims_%d_resolution_%d_%s.mat" % (elem_dims, dims, resolution, elem_fnames[elem_dims])
            fname = os.path.join(test_data_dir, fname)
            assert(os.path.isfile(fname)) #If you fail here, the generation of test data using generate_test_data.py failed or was not performed

            data = sio.loadmat(fname)
            points, elems, D = data["points"], data["elems"], data["D"]
            nr_points = points.shape[0]

            #D specified at initialization
            solver = FIMPY.create_fim_solver(points, elems, D, precision=precision, device=device, use_active_list=use_active_list)
            x0 = np.array([np.random.choice(nr_points)])
            x0_vals = np.array([0.])
            phi1 = solver.comp_fim(x0, x0_vals)
            assert(phi1.dtype == precision)
            assert(phi1.ndim == 1)
            assert(phi1.size == nr_points)
            assert(np.all(~np.isnan(phi1)))

            #D specified at computation
            solver = FIMPY.create_fim_solver(points, elems, precision=precision, device=device, use_active_list=use_active_list)
            #Fails without specifying D
            with pytest.raises(Exception):
                solver.comp_fim(x0, x0_vals)

            phi2 = solver.comp_fim(x0, x0_vals, D)
            assert(phi2.dtype == precision)
            assert(phi2.ndim == 1)
            assert(phi2.size == nr_points)
            assert(np.all(~np.isnan(phi2)))

            #Results should be the same
            assert(np.allclose(phi1, phi2))




    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('dims', test_dims)
    @pytest.mark.parametrize('elem_dims', test_elem_dims)
    @pytest.mark.parametrize('use_active_list', [True, False])
    def test_comp_gpu(self, dims, elem_dims, precision, use_active_list):
        self.test_comp(dims, elem_dims, precision, use_active_list=use_active_list, device='gpu')
