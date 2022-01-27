from fimpy.utils.comp import metric_norm_matrix_2D_cython, metric_norm_matrix_3D_cython
import numpy as np
import pytest

class TestCythonMethods():

    @pytest.mark.parametrize("prec", [np.float32, np.float64])
    def test_2D_vec(self, prec):
        rtol = (1e-3 if prec == np.float32 else 1e-4)
        A = np.random.normal(size=(100, 2, 2)).astype(prec)
        A[..., 1, 0] = A[..., 0, 1] #Symmetrize

        z1 = np.random.normal(size=(100, 2)).astype(prec)
        z2 = np.random.normal(size=(100, 2)).astype(prec)

        expected_result = np.einsum('...x,...xy,...y->...', z1, A, z2)
        comp_result = metric_norm_matrix_2D_cython(A, z1, z2, ret_sqrt=False)
        assert np.allclose(expected_result, comp_result, rtol=rtol)

        #Broadcasted
        A = np.random.normal(size=(100, 1, 2, 2))
        A[..., 1, 0] = A[..., 0, 1] #Symmetrize

        z1 = np.random.normal(size=(100, 3, 2))
        z2 = np.random.normal(size=(100, 3, 2))

        expected_result = np.einsum('...x,...xy,...y->...', z1, A, z2)
        comp_result = metric_norm_matrix_2D_cython(A, z1, z2, ret_sqrt=False)
        assert np.allclose(expected_result, comp_result, rtol=rtol) #Broadcasted
        
    @pytest.mark.parametrize("prec", [np.float32, np.float64])
    def test_3D_vec(self, prec):
        rtol = (1e-3 if prec == np.float32 else 1e-4)
        A = np.random.normal(size=(100, 3, 3)).astype(prec)
        A[..., 1, 0] = A[..., 0, 1] #Symmetrize
        A[..., 2, 0] = A[..., 0, 2] 
        A[..., 2, 1] = A[..., 1, 2]

        z1 = np.random.normal(size=(100, 3)).astype(prec)
        z2 = np.random.normal(size=(100, 3)).astype(prec)

        expected_result = np.einsum('...x,...xy,...y->...', z1, A, z2)
        comp_result = metric_norm_matrix_3D_cython(A, z1, z2, ret_sqrt=False)
        assert np.allclose(expected_result, comp_result, rtol=rtol) #Single vectorized

        #Broadcasted
        A = np.random.normal(size=(100, 1, 3, 3))
        A[..., 1, 0] = A[..., 0, 1] #Symmetrize
        A[..., 2, 0] = A[..., 0, 2] 
        A[..., 2, 1] = A[..., 1, 2]

        z1 = np.random.normal(size=(100, 4, 3))
        z2 = np.random.normal(size=(100, 4, 3))

        expected_result = np.einsum('...x,...xy,...y->...', z1, A, z2)
        comp_result = metric_norm_matrix_3D_cython(A, z1, z2, ret_sqrt=False)
        assert np.allclose(expected_result, comp_result, rtol=rtol)
        

