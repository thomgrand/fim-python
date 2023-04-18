"""This file contains the interface to create the Fast Iterative Method solvers.
"""

try:
    import cupy as cp
    from .fim_cupy import FIMCupyAL, FIMCupy
    cupy_available = True
except ImportError as err:
    cupy_available = False
    print("Import of Cupy failed. The GPU version of fimpy will be unavailable. Message: %s" % (err))

from .fim_base import FIMBase
from .fim_np import FIMNP, FIMNPAL
import numpy as np
from typing import Union

available_arr_t = (Union[np.ndarray, cp.ndarray] if cupy_available else np.ndarray)

"""Function responsible for creating the correct Fast Iterative Method solver
"""
def create_fim_solver(points : available_arr_t, elems : available_arr_t, metrics : available_arr_t =None, 
                    precision=np.float32, device='gpu', use_active_list=True) -> FIMBase:
    """Creates a Fast Iterative Method solver for solving the anisotropic eikonal equation

    .. math::
        \\left\\{
        \\begin{array}{rll}
        \\left<\\nabla \\phi, D \\nabla \\phi \\right> &= 1 \\quad &\\text{on} \\; \\Omega \\\\
        \\phi(\\mathbf{x}_0) &= g(\\mathbf{x}_0) \\quad &\\text{on} \\; \\Gamma
        \\end{array}
        \\right. .

    Parameters
    ----------
    points : Union[np.ndarray (float), cp.ndarray (float)]
        Array of points, :math:`n \\times d`
    elems : Union[np.ndarray (int), cp.ndarray (int)]
        Array of elements, :math:`m \\times d_e`
    metrics : Union[np.ndarray (float), cp.ndarray (float)], optional
        Specifies the initial :math:`D \\in \\mathbb{R}^{d \\times d}` tensors.
        If not specified, you later need to provide them in :meth:`comp_fim <fimpy.fim_base.FIMBase.comp_fim>`, by default None
    precision : np.dtype, optional
        precision of all calculations and the final result, by default np.float32
    device : str, optional
        Specifies the target device for the computations. One of [cpu, gpu], by default 'gpu'
    use_active_list : bool, optional
        If set to true, you will get an active list solver that only computes the necessary subset of points in each iteration.
        If set to false, a Jacobi solver will be returned that updates all points of the mesh in each iteration. By default True

    Returns
    -------
    FIMBase
        Returns a Fast Iterative Method solver
    """
    if not cupy_available:
        device='cpu'

    if device == 'cpu':
        return (FIMNPAL(points, elems, metrics, precision) if use_active_list else FIMNP(points, elems, metrics, precision))
    elif device == 'gpu':
        return (FIMCupyAL(points, elems, metrics, precision) if use_active_list else FIMCupy(points, elems, metrics, precision))
