"""This file contains the norm_map, that is in general used to efficiently select the best and fastest function to compute norms of the type :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>`.
"""
#import utility
import numpy as np
from .comp import metric_norm_matrix_2D_cython, metric_norm_matrix_3D_cython
from .comp import metric_sqr_norm_matrix, metric_norm_matrix
from collections import defaultdict

norm_map = {np: defaultdict(lambda: (metric_norm_matrix, metric_sqr_norm_matrix))}
norm_map[np][2] = (lambda A, x1, x2: metric_norm_matrix_2D_cython(A, x1, x2, True), lambda A, x1, x2: metric_norm_matrix_2D_cython(A, x1, x2, False))
norm_map[np][3] = (lambda A, x1, x2: metric_norm_matrix_3D_cython(A, x1, x2, True), lambda A, x1, x2: metric_norm_matrix_3D_cython(A, x1, x2, False))

try:
    import cupy as cp
    from .comp import metric_norm_matrix_2D_cupy, metric_norm_matrix_3D_cupy, metric_sqr_norm_matrix_2D_cupy, metric_sqr_norm_matrix_3D_cupy
    norm_map[cp] = defaultdict(lambda: (lambda A, x1, x2: metric_norm_matrix(A, x1, x2, cp), lambda A, x1, x2: metric_sqr_norm_matrix(A, x1, x2, cp)))
    norm_map[cp][2] = (lambda A, x1, x2: metric_norm_matrix_2D_cupy(A, x1, x2), lambda A, x1, x2: metric_sqr_norm_matrix_2D_cupy(A, x1, x2))
    norm_map[cp][3] = (lambda A, x1, x2: metric_norm_matrix_3D_cupy(A, x1, x2), lambda A, x1, x2: metric_sqr_norm_matrix_3D_cupy(A, x1, x2))
except ImportError as err:
    ...
