"""This file contains small custom functions to compute :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>` for general and special dimensions :math:`d`.
Note that all custom, efficient implementations assume that :math:`A` is a symmetric matrix.
"""
import numpy as np
from .cython.comp import metric_sqr_norm_matrix_2D_vec, metric_sqr_norm_matrix_3D_vec
try:
  import cupy as cp
  cupy_enabled = True
except ImportError as err:
  cupy_enabled = False


def _broadcast_metric_params(A, x1, x2):
  if A.ndim == 4: #Broadcasting
    #To remove the read-only problem, copy back
    A = np.broadcast_to(A, [A.shape[0], max(A.shape[1], x1.shape[1]), x1.shape[-1], x1.shape[-1]]).copy()
    x1 = np.broadcast_to(x1, A.shape[:-1]).copy()
    x2 = np.broadcast_to(x2, A.shape[:-1]).copy()

  return A, x1, x2


def metric_norm_matrix_2D_cython(A, x1, x2, ret_sqrt=True):
  A, x1, x2 = _broadcast_metric_params(A, x1, x2)
  A_flat, x1_flat, x2_flat = A.reshape([-1, 2, 2]), x1.reshape([-1, 2]), x2.reshape([-1, 2])
  norm = metric_sqr_norm_matrix_2D_vec(A_flat, x1_flat, x2_flat)

  if ret_sqrt:
    norm = np.sqrt(norm)

  return norm.reshape(A.shape[:-2])

def metric_norm_matrix_3D_cython(A, x1, x2, ret_sqrt=True):
  A, x1, x2 = _broadcast_metric_params(A, x1, x2)
  A_flat, x1_flat, x2_flat = A.reshape([-1, 3, 3]), x1.reshape([-1, 3]), x2.reshape([-1, 3])
  norm = metric_sqr_norm_matrix_3D_vec(A_flat, x1_flat, x2_flat)

  if ret_sqrt:
    norm = np.sqrt(norm)

  return norm.reshape(A.shape[:-2])


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
