"""This file contains small custom functions to compute :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>` for general and special dimensions :math:`d`.
Note that all custom, efficient implementations assume that :math:`A` is a symmetric matrix.
"""
import numpy as np
from numba import njit, prange, vectorize, guvectorize
import numba
try:
  import cupy as cp
  cupy_enabled = True
except ImportError as err:
  cupy_enabled = False

@njit(cache=True, nogil=True) #, parallel=True)
def metric_norm_matrix_2D_njit(A, x1, x2, ret_sqrt=True):
  """Custom implementation of :func:`metric_norm_matrix` for ``lib = np`` and :math:`d = 2`.
  """
  a, b, c = A[..., 0, 0], A[..., 1, 1], A[..., 0, 1]
  norm = (x1[..., 0] * (a * x2[..., 0] + c * x2[..., 1]) + x1[..., 1] * (c * x2[..., 0] + b * x2[..., 1]))

  if ret_sqrt:
    norm = np.sqrt(norm)

  return norm

@vectorize([numba.float32(*(numba.float32,)*12),
            numba.float64(*(numba.float64,)*12)], cache=True, target='cpu')
def metric_sqr_norm_matrix_3D_vec(a, b, c, d, e, f, x11, x12, x13, x21, x22, x23):
  return (x11 * (a * x21 + d * x22 + e * x23)
          + x12 * (d * x21 + b * x22 + f * x23)
          + x13 * (e * x21 + f * x22 + c * x23))

@vectorize([numba.float32(*(numba.float32,)*12),
            numba.float64(*(numba.float64,)*12)], cache=True, target='cpu')
def metric_norm_matrix_3D_vec(a, b, c, d, e, f, x11, x12, x13, x21, x22, x23):
  return np.sqrt(x11 * (a * x21 + d * x22 + e * x23)
          + x12 * (d * x21 + b * x22 + f * x23)
          + x13 * (e * x21 + f * x22 + c * x23))

@njit(cache=True, nogil=True, parallel=True)
def metric_norm_matrix_3D_njit_flat(A, x1, x2, ret_sqrt=True):
  """Custom implementation of :func:`metric_norm_matrix` for an [?, d, d] array, ``lib = np`` and :math:`d = 3`.
  """
  #return metric_norm_matrix(A, x1, x2)
  a, b, c, d, e, f = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2], A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]
  #norm = (x1[..., 0] * (a * x2[..., 0] + d * x2[..., 1] + e * x2[..., 2])
  #        + x1[..., 1] * (d * x2[..., 0] + b * x2[..., 1] + f * x2[..., 2])
  #        + x1[..., 2] * (e * x2[..., 0] + f * x2[..., 1] + c * x2[..., 2]))
  if ret_sqrt:
    norm = metric_norm_matrix_3D_vec(a, b, c, d, e, f, 
                                x1[..., 0], x1[..., 1], x1[..., 2],
                                x2[..., 0], x2[..., 1], x2[..., 2])
  else:
    norm = metric_sqr_norm_matrix_3D_vec(a, b, c, d, e, f, 
                                x1[..., 0], x1[..., 1], x1[..., 2],
                                x2[..., 0], x2[..., 1], x2[..., 2])

  return norm

@njit(cache=True, nogil=True, parallel=True)
def metric_norm_matrix_3D_njit_bc(A, x1, x2, ret_sqrt=True):
  """Custom implementation of :func:`metric_norm_matrix` for an [?, ?, d, d] array, ``lib = np`` and :math:`d = 3`.
  """
  a, b, c, d, e, f = A[..., 0, 0, 0], A[..., 0, 1, 1], A[..., 0, 2, 2], A[..., 0, 0, 1], A[..., 0, 0, 2], A[..., 0, 1, 2]
  norm = np.empty(shape=(x1.shape[0], x1.shape[1]), dtype=x1.dtype)
  if ret_sqrt:
    for i in prange(x1.shape[1]):
      norm[:, i] = metric_norm_matrix_3D_vec(a, b, c, d, e, f, 
                                                  x1[..., i, 0], x1[..., i, 1], x1[..., i, 2],
                                                  x2[..., i, 0], x2[..., i, 1], x2[..., i, 2])
  else:
    for i in prange(x1.shape[1]):
      norm[:, i] = metric_sqr_norm_matrix_3D_vec(a, b, c, d, e, f, 
                                                  x1[..., i, 0], x1[..., i, 1], x1[..., i, 2],
                                                  x2[..., i, 0], x2[..., i, 1], x2[..., i, 2])

  return norm

@njit
def metric_norm_matrix_3D_njit(A, x1, x2, ret_sqrt=True):
  """Custom implementation of :func:`metric_norm_matrix` for ``lib = np`` and :math:`d = 3`.
  """
  if A.ndim == 4:
    if A.shape[1] == 1:
      return metric_norm_matrix_3D_njit_bc(A, x1, x2, ret_sqrt)
    else:
      dims = A.shape[-1]
      return metric_norm_matrix_3D_njit_flat(A.reshape((-1, dims, dims)), x1.reshape((-1, dims)), x2.reshape((-1, dims)), ret_sqrt).reshape((A.shape[0], A.shape[1]))
      
  else:
    return metric_norm_matrix_3D_njit_flat(A, x1, x2, ret_sqrt)

if cupy_enabled:
  @cp.fuse()
  def metric_sqr_norm_matrix_2D_cupy(A, x1, x2):
    """Custom implementation of :func:`metric_sqr_norm_matrix` for ``lib = cp`` and :math:`d = 3`.
    """
    a, b, c = A[..., 0, 0], A[..., 1, 1], A[..., 0, 1]
    return (x1[..., 0] * (a * x2[..., 0] + c * x2[..., 1]) + x1[..., 1] * (c * x2[..., 0] + b * x2[..., 1]))

  @cp.fuse()
  def metric_norm_matrix_2D_cupy(A, x1, x2):
    """Custom implementation of :func:`metric_norm_matrix` for ``lib = cp`` and :math:`d = 3`.
    """
    norm = metric_sqr_norm_matrix_2D_cupy(A, x1, x2)
    norm = cp.sqrt(norm)

    return norm

  @cp.fuse()
  def metric_sqr_norm_matrix_3D_cupy(A, x1, x2):
    """Custom implementation of :func:`metric_sqr_norm_matrix` for ``lib = cp`` and :math:`d = 3`.
    """
    a, b, c, d, e, f = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2], A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]
    norm = (x1[..., 0] * (a * x2[..., 0] + d * x2[..., 1] + e * x2[..., 2])
            + x1[..., 1] * (d * x2[..., 0] + b * x2[..., 1] + f * x2[..., 2])
            + x1[..., 2] * (e * x2[..., 0] + f * x2[..., 1] + c * x2[..., 2]))

    return norm

  @cp.fuse()
  def metric_norm_matrix_3D_cupy(A, x1, x2):
    """Custom implementation of :func:`metric_norm_matrix` for ``lib = cp`` and :math:`d = 3`.
    """
    norm = metric_sqr_norm_matrix_3D_cupy(A, x1, x2)
    norm = cp.sqrt(norm)

    return norm


def metric_sqr_norm_matrix(A, x1, x2, lib=np):
  """Computes :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2\\right>` in a broadcasted fashion for arbitrary :math:`d`.
  Details on the parameters and the return value can be found in :func:`metric_norm_matrix`.
  """
  #assert(A.shape[-1] == x1.shape[-1] and A.shape[-2] == A.shape[-1])

  #assert(np.all(np.equal(x2.shape[-1], x1.shape[-1])))

  sqr_norm = lib.sum(lib.sum(A * x1[..., lib.newaxis], axis=-2) * x2, axis=-1)
  return sqr_norm

def metric_norm_matrix(A, x1, x2, lib=np):
  """Computes :math:`\\sqrt{\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>}` in a broadcasted fashion for arbitrary :math:`d`

  Parameters
  ----------
  A : ndarray (float)
      An [..., d, d] array with the stack of matrices :math:`A`
  x1 : ndarray (float)
      An [..., d] array with the stack of vectors :math:`\\mathbf{x}_1`
  x2 : ndarray (float)
      An [..., d] array with the stack of vectors :math:`\\mathbf{x}_2`
  lib : library, optional
      Library used for the computations of the norm. Needs to implement a ``sum`` method.
      By default np

  Returns
  -------
  ndarray (float)
      An [...] array holding the computed norms :math:`\\sqrt{\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>}`.
  """

  
  #assert(A.shape[-1] == x1.shape[-1] and A.shape[-2] == A.shape[-1])

  #assert(np.all(np.equal(x2.shape[-1], x1.shape[-1])))

  sqr_norm = metric_sqr_norm_matrix(A, x1, x2, lib)
  return lib.sqrt(sqr_norm)
