# cython: infer_types=True
# distutils: language=c++
cimport cython
from cython.parallel cimport prange
from libcpp.vector cimport vector
from typing import Callable
from libcpp cimport bool as bool_t
from libc.math cimport sqrt

import numpy as np

ctypedef fused mat_t:
    float
    double

  
ctypedef fused ind_t:
    int
    long long


@cython.boundscheck(False)
@cython.wraparound(False)
def metric_sqr_norm_matrix_2D_vec(A_np, mat_t[:, :] x1, mat_t[:, :] x2):
  result = np.empty(shape=A_np.shape[0], dtype=A_np.dtype)
  cdef mat_t[:, :, :] A = A_np
  a, b, c = A[..., 0, 0], A[..., 1, 1], A[..., 0, 1]

  
  cdef mat_t[::1] result_view = result
  cdef mat_t[:] a_ = a
  cdef mat_t[:] b_ = b
  cdef mat_t[:] c_ = c
  cdef int rows, row_i

  rows = A.shape[0]
  for row_i in prange(rows, nogil=True):
    result_view[row_i] = (x1[row_i, 0] * (a_[row_i] * x2[row_i, 0] + c_[row_i] * x2[row_i, 1]) 
                            + x1[row_i, 1] * (c_[row_i] * x2[row_i, 0] + b_[row_i] * x2[row_i, 1]))


  return result

@cython.boundscheck(False)
@cython.wraparound(False)
def metric_sqr_norm_matrix_3D_vec(A_np, mat_t[:, :] x1, mat_t[:, :] x2):
  result = np.empty(shape=A_np.shape[0], dtype=A_np.dtype)
  cdef mat_t[:, :, :] A = A_np
  a, b, c, d, e, f = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2], A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]

  cdef mat_t[::1] result_view = result
  cdef int rows, row_i
  cdef mat_t[:] a_ = a
  cdef mat_t[:] b_ = b
  cdef mat_t[:] c_ = c
  cdef mat_t[:] d_ = d
  cdef mat_t[:] e_ = e
  cdef mat_t[:] f_ = f

  cdef mat_t[:] x11_ = x1[..., 0]
  cdef mat_t[:] x12_ = x1[..., 1]
  cdef mat_t[:] x13_ = x1[..., 2]
  cdef mat_t[:] x21_ = x2[..., 0]
  cdef mat_t[:] x22_ = x2[..., 1]
  cdef mat_t[:] x23_ = x2[..., 2]

  rows = A.shape[0]
  for row_i in prange(rows, nogil=True):
    result_view[row_i] = (x11_[row_i] * (a_[row_i] * x21_[row_i] + d_[row_i] * x22_[row_i] + e_[row_i] * x23_[row_i])
                        + x12_[row_i] * (d_[row_i] * x21_[row_i] + b_[row_i] * x22_[row_i] + f_[row_i] * x23_[row_i])
                        + x13_[row_i] * (e_[row_i] * x21_[row_i] + f_[row_i] * x22_[row_i] + c_[row_i] * x23_[row_i]))


  return result
