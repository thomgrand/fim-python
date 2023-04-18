"""This file contains the base implementation of the Fast Iterative Method, common to all solvers.
"""

from typing import Type
import numpy as np
from itertools import permutations
from abc import abstractmethod
from .utils.tsitsiklis import norm_map
from .fim_cutils import compute_point_elem_map_c, compute_neighborhood_map_c

class FIMBase():
    """This abstract base class combines common functionality of Cupy and Numpy solvers

    Parameters
    ----------
    points : Union[np.ndarray (float), cp.ndarray (float)]
        Array of points, :math:`n \\times d`
    elems : Union[np.ndarray (int), cp.ndarray (int)]
        Array of elements, :math:`m \\times d_e`
    metrics : Union[np.ndarray (float), cp.ndarray (float)], optional
        Specifies the initial :math:`D \\in \\mathbb{R}^{d \\times d}` tensors.
        If not specified, you later need to provide them in :meth:`comp_fim`, by default None
    precision : np.dtype, optional
        precision of all calculations and the final result, by default np.float32
    comp_connectivities : bool, optional
        If set to true, both a neighborhood and point to element mapping will be computed. 
        These are used to efficiently compute the active list.
        By default False
    """

    undef_val = 1e10 #: The value for points that have not been computed yet
    convergence_eps = 1e-9 #: The value of epsilon to check if a point has converged

    def __init__(self, points, elems, metrics=None, precision=np.float32, comp_connectivities=False):
        """
        """
        assert(np.issubdtype(points.dtype, np.floating))
        assert(np.issubdtype(elems.dtype, np.integer))

        assert(points.ndim == 2)
        #assert(points.shape[-1] in [1, 2, 3]) #TODO: Necessary?
        assert(elems.ndim == 2)
        assert(elems.shape[-1] in [2, 3, 4])
        elems = elems.astype(np.int32)
        
        self.nr_points = points.shape[0]
        self.nr_elems = elems.shape[0]
        assert(np.unique(elems).size == self.nr_points) #All points are part of at least one element
        assert(np.all(np.unique(elems) == np.arange(self.nr_points))) #All points are part of at least one element

        self.points = points.astype(precision)
        self.elems = np.ascontiguousarray(elems)


        if metrics is not None:        
            self.check_metrics_argument(metrics)
            metrics = np.linalg.inv(metrics).astype(precision) #The inverse metric is used in the FIM algorithm

        self.metrics = metrics

        #General allocations
        self.active_list = np.zeros(shape=[self.nr_points], dtype=bool)
        self.elems_perm = self.compute_unique_elem_permutations()
        self.points_perm = self.points[self.elems_perm]
        self.phi_sol = np.ones_like(self.points[..., 0]) * self.undef_val

        self.elem_dims = self.elems.shape[-1]
        if comp_connectivities:
            self.nh_map = self.compute_neighborhood_map()
            self.point_elem_map = self.compute_point_elem_map()

        self.precision = precision
        self.norm_map = norm_map
        self.dims = self.points.shape[-1]
        self.choose_update_alg()

    def choose_update_alg(self):
        """Selects the update step of the algorithm according to the provided element type.

        Raises
        ------
        TypeError
            If self.elem_dims is not in [2, 3, 4] (lines, triangles, tetrahedra)
        """
        if self.elem_dims == 2:
            self.update_all_points = self.calculate_all_line_updates
            self.update_specific_points = self.calculate_specific_line_updates
        elif self.elem_dims == 3:
            self.update_all_points = self.calculate_all_triang_updates
            self.update_specific_points = self.calculate_specific_triang_updates
        elif self.elem_dims == 4:
            self.update_all_points = self.calculate_all_tetra_updates
            self.update_specific_points = self.calculate_specific_tetra_updates
        else:
            raise TypeError("Unsupported number of points per element: %d. Supported are lines, triangles and tetrahedra (2, 3, 4)" % (self.elem_dims))

    def check_metrics_argument(self, metrics):
        """Checks the validity of the metric tensors (:math:`D \\in \\mathbb{R}^{d \\times d}`)

        Parameters
        ----------
        metrics : Union[np.ndarray (float), cp.ndarray (float)], optional
            The :math:`D \\in \\mathbb{R}^{d \\times d}` tensors.
        """
        assert(np.issubdtype(metrics.dtype, np.floating))
        assert(metrics.shape[0] == self.nr_elems) #One constant metric for each element
        assert(metrics.ndim == 3)
        assert(metrics.shape[-1] == metrics.shape[-2] and metrics.shape[-1] == self.points.shape[-1])
        assert(np.allclose(metrics - np.transpose(metrics, axes=(0, 2, 1)), 0.)) #Symmetric
        assert(np.all(np.linalg.eigh(metrics)[0] > 1e-4)) #Positive definite


    def compute_unique_elem_permutations(self):
        """Returns all point permutations of each element (i.e. :math:`[M, d_e] \to [M, d_e, d_e]`).

        Returns
        -------
        ndarray (int)
            An [M, d_e, d_e] array containing all permutations.
        """
        if self.elems.shape[1] == 2: #Lines
            perms = np.array([[0, 1], [1, 0]])
        elif self.elems.shape[1] == 3: #Triangles
            perms = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]])
        elif self.elems.shape[1] == 4: #Tetrahedra
            perms = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]])
            
        elems_perm = np.stack([self.elems[np.arange(self.nr_elems)[..., np.newaxis], perm[np.newaxis]] for perm in perms], axis=1)
        return elems_perm

    def compute_neighborhood_map(self):
        """Computes the neighborhood map for the given mesh.

        Returns
        -------
        ndarray (int)
            The [N, ?] array holding all neighbors for each point.
            shape[1] of the return value will be equal to the maximum neighbor connectivity.
        """
        max_point_elem_ratio = np.max(np.unique(self.elems, return_counts=True)[1])
        nh_map = np.zeros(shape=[self.nr_points, max_point_elem_ratio * self.elem_dims], dtype=np.int32)

        nh_map = np.array(compute_neighborhood_map_c(self.elems, nh_map))

        nh_map = np.sort(nh_map, axis=-1)
        
        # There may be cases where the ratio was an overestimate
        while nh_map.shape[1] > 1 and np.all(nh_map[..., -1] == nh_map[..., -2]):
            nh_map = nh_map[..., :-1]

        return nh_map

    def compute_point_elem_map(self):
        """Computes the point element mapping for each point.

        Returns
        -------
        ndarray (int)
            The [N, ?] array holding for each point all elements that it is contained in.
            shape[1] of the return value will be equal to the maximum point to element ratio.
        """
        max_point_elem_ratio = np.max(np.unique(self.elems, return_counts=True)[1])
        point_elem_map = np.zeros(shape=[self.nr_points, max_point_elem_ratio], dtype=np.int32)
        point_elem_map = np.sort(compute_point_elem_map_c(self.elems, point_elem_map), axis=-1)
        return point_elem_map

    def tsitsiklis_update_line(self, x1, x2, D, u1, lib=np):
        """Computes :math:`||\\mathbf{x}_2 - \\mathbf{x}_1||_M` in a broadcasted way.

        Parameters
        ----------
        x1 : ndarray (precision)
            An [..., N, d] array holding :math:`\\mathbf{x}_1`
        x2 : ndarray (precision)
            An [..., N, d] array holding :math:`\\mathbf{x}_2`
        D : ndarray (precision)
            An [..., N, d, d] array holding :math:`D`
        u1 : ndarray (precision)
            An [..., N] array holding :math:`u_1`
        lib : Union([np, cp]), optional
            Module that will be used to compute the norm, by default np

        Returns
        -------
        ndarray (precision)
            An [..., N] array that holds :math:`||\\mathbf{x}_2 - \\mathbf{x}_1||_M`
        """
        norm_f = self.norm_map[lib][D.shape[-1]][0]
        a1 = x2 - x1
        return u1 + norm_f(D, a1, a1)
    
    def tsitsiklis_update_point_sol(self, x1, x2, x3, D, u1, u2, lib=np):
        """Computes 
        
        .. math::
            \\min\\{u_1 + ||\\mathbf{x}_3 - \\mathbf{x}_1||_M, u_2 + ||\\mathbf{x}_3 - \\mathbf{x}_2||_M\\}        
        
        in a broadcasted way.
        For more information on the type and shape of the parameters and return value, see :meth:`tsitsiklis_update_line`.
        """
        norm_f = self.norm_map[lib][D.shape[-1]][0]

        a1 = x3 - x1
        a2 = x3 - x2
        u3_1 = u1 + norm_f(D, a1, a1)
        u3_2 = u2 + norm_f(D, a2, a2)

        return lib.minimum(u3_1, u3_2)

    def tsitsiklis_update_triang(self, x1, x2, x3, D, u1, u2, lib=np):
        """Computes the update inside a single triangle as 

        .. math::
            \\min_{\\lambda} \\lambda u_1 + (1 - \\lambda) u_2 + ||\\mathbf{x}_3 - (\\lambda \\mathbf{x}_1 + (1 - \\lambda) \\mathbf{x}_2)||_M

        For more information on the type and shape of the parameters and return value, see :meth:`tsitsiklis_update_line`.
        """
        k = u1 - u2
        z2 = x2 - x3
        z1 = x1 - x2

        norm_f, norm_sqr_f = self.norm_map[lib][D.shape[-1]]

        p11 = norm_sqr_f(D, x1=z1, x2=z1)
        p12 = norm_sqr_f(D, x1=z1, x2=z2)
        p22 = norm_sqr_f(D, x1=z2, x2=z2)
        denominator = p11 - k**2
        sqrt_val = (p11 * p22 - p12**2) / denominator
        sqrt_invalid_mask = sqrt_val < 0.
        sqrt_op = lib.sqrt(sqrt_val)
        rhs = k * sqrt_op
        alpha1 = -(p12 + rhs) / p11
        alpha2 = -(p12 - rhs) / p11
        alpha1 = lib.minimum(lib.maximum(alpha1, 0.), 1.)
        alpha2 = lib.minimum(lib.maximum(alpha2, 0.), 1.)

        u3 = []
        for alpha in [alpha1, alpha2]:
            x = x3 - (alpha[..., lib.newaxis] * x1 + (1 - alpha[..., lib.newaxis]) * x2)
            u3.append(alpha * u1 + (1 - alpha) * u2
                    + norm_f(D, x, x))

        u3 = lib.minimum(*u3)
        u3_point = self.tsitsiklis_update_point_sol(x1, x2, x3, D, u1, u2, lib=lib)
        u3_computed = lib.where(sqrt_invalid_mask, u3_point, u3)
        u3_final = u3_computed

        return u3_final

    def tsitsiklis_update_tetra_quadr(self, D, k, z1, z2, lib=np):
        """Computes the quadratic equation for the tetrahedra update in :meth:`calculate_tet_update`.
        """
        norm_f, norm_sqr_f = self.norm_map[lib][D.shape[-1]]
        p11 = norm_sqr_f(D, z1, z1)
        p12 = norm_sqr_f(D, z1, z2)
        p22 = norm_sqr_f(D, z2, z2)
        denominator = p11 - k*k
        sqrt_val = (p11 * p22 - (p12 * p12)) / denominator
        rhs = k * lib.sqrt(sqrt_val)
        alpha1 = -(p12 + rhs) / p11
        #alpha2 = -(p12 - rhs) / p11

        return alpha1 #, alpha2

    def tsitsiklis_update_tetra(self, x1, x2, x3, x4, D, u1, u2, u3, lib=np):
        """Computes the update inside a single tetrahedron by computing 

            - the update inside the tetrahedron (:meth:`calculate_tet_update`)
            - the update on each face (:meth:`tsitsiklis_update_triang`)

            and taking the minimum across all possible values.
        For more information on the type and shape of the parameters and return value, see :meth:`tsitsiklis_update_line`.
        """
        u_tet = self.calculate_tet_update(x1, x2, x3, x4, D, u1, u2, u3, lib)
        u_tet = lib.where(lib.isnan(u_tet), lib.inf, u_tet)

        #Face calculations (Includes possible line calculations)
        u_triang = lib.minimum(self.tsitsiklis_update_triang(x1, x2, x4, D, u1, u2, lib),
                               self.tsitsiklis_update_triang(x1, x3, x4, D, u1, u3, lib))
        u_triang = lib.minimum(u_triang,
                               self.tsitsiklis_update_triang(x2, x3, x4, D, u2, u3, lib))

        return lib.minimum(u_triang, u_tet)

    def calculate_tet_update(self, x1, x2, x3, x4, D, u1, u2, u3, lib=np):
        """Computes the update inside a single tetrahedron as 

        .. math::
            \\min_{\\lambda_1, \\lambda_2} \\lambda_1 u_1 + \\lambda_2 u_2 + \\lambda_3 u_3 + ||\\mathbf{x}_4 - \\mathbf{x}_{1, 2, 3}||_M

        for :math:`\\lambda_3 = 1 - \\lambda_1 - \\lambda_2` and :math:`\\mathbf{x}_{1, 2, 3} = \\lambda_1 \\mathbf{x}_1 + \\lambda_2 \\mathbf{x}_2 + \\lambda_3 \\mathbf{x}_3`.
        For more information on the type and shape of the parameters and return value, see :meth:`tsitsiklis_update_line`.
        """
        xs = lib.stack([x1, x2, x3, x4], axis=-1)
        us = lib.stack([u1, u2, u3], axis=-1)
        norm_f, norm_sqr_f = self.norm_map[lib][D.shape[-1]]

        y3 = x4 - x3
        y1 = x3 - x1
        y2 = x3 - x2

        k1 = u1 - u3
        k2 = u2 - u3
        #k3 = u3

        r11 = norm_sqr_f(D, y1, y1)
        r12 = norm_sqr_f(D, y1, y2)
        r13 = norm_sqr_f(D, y1, y3)
        r21 = r12
        r22 = norm_sqr_f(D, y2, y2)
        r23 = norm_sqr_f(D, y2, y3)
        r31 = r13
        r32 = r23
        #r33 = norm_sqr_f(D, y3, y3)    

        A1 = k2 * r11 - k1 * r12
        A2 = k2 * r21 - k1 * r22
        B = k2 * r31 - k1 * r32
        k = k1 - A1 / A2 * k2
        #u = k3 - B / A2 * k2
        z1 = y1 - (A1 /A2)[..., lib.newaxis] * y2
        z2 = y3 - (B / A2)[..., lib.newaxis] * y2

        alpha1 = self.tsitsiklis_update_tetra_quadr(D, k, z1, z2, lib)
        alpha2 = -(B + alpha1 * A1) / A2

        special_case1 = ((A1 == 0) & (A2 == 0))
        alpha1 = lib.where(special_case1, (r12 * r23 - r13 * r22) / (r11 * r22 - (r12 * r12)), alpha1)
        alpha2 = lib.where(special_case1, (r12 * r13 - r11 * r23) / (r11 * r22 - (r12 * r12)), alpha2)

        special_case2 = ((A1 == 0) & ~special_case1)
        alpha1 = lib.where(special_case2, 0, alpha1)
        alpha2 = lib.where(special_case2, -B / A2, alpha2)

        special_case3 = ((A2 == 0) & ~special_case1 & ~special_case2)
        alpha1 = lib.where(special_case3, -B / A1, alpha1)
        alpha2 = lib.where(special_case3, 0, alpha2)

        alphas = lib.stack([alpha1, alpha2, 1 - alpha1 - alpha2], axis=-1)
        if alphas.ndim == xs.ndim - 1:
            alphas = alphas[..., lib.newaxis, :]

        dist = x4 - lib.sum(xs[..., :-1] * alphas, axis=-1) #dist = x4 - (alpha1 * x1 + alpha2 * x2 + alpha3 * x3)
        alphas = lib.squeeze(alphas)
        return lib.where(lib.any((alphas < 0) | (alphas > 1), axis=-1), lib.inf, norm_f(D, dist, dist) + lib.sum(alphas * us, axis=-1)) #(alpha1 * u1 + alpha2 * u2 + alpha3 * u3)



    def calculate_specific_triang_updates(self, elems_perm, xs_perm, D, us, lib=np):
        us_new = us.copy()

        #perms = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]])
        us_perm = us[elems_perm]

        us_result = self.tsitsiklis_update_triang(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :],
                                                  D, us_perm[..., 0], us_perm[..., 1], lib=lib)

        #Now we need to take the minimum result of old and all new
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new

    def calculate_specific_line_updates(self, elems_perm, xs_perm, D, us, lib=np):
        us_new = us.copy() 
        us_perm = us[elems_perm]
        us_result = self.tsitsiklis_update_line(xs_perm[..., 0, :], xs_perm[..., 1, :],
                                D, us_perm[..., 0], lib=lib)
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new

    def calculate_all_line_updates(self, elems_perm, xs_perm, D, us, lib=np):
        """Calculates all lines updates for all element permutations and computes their minimum as the new solution of :math:`\\phi`.

        Parameters
        ----------
        elems_perm : ndarray (int)
            All point permutations obtained using :meth:`compute_unique_elem_permutations`
        xs_perm : ndarray (float)
            All point coordinate permutations as an :math:`[M, d_e, d_e, d]` array.
        D : ndarray (float)
            An [M, d, d] array containing :math:`D`.
        us : ndarray (float)
            An [N] array containing the current solution of :math:`\\phi_k`.
        lib : library, optional
            Library to use for the computations, by default np

        Returns
        -------
        ndarray (float)
            An [N] array holding the new solution :math:`\\phi_{k+1}`.
        """
        us_new = us.copy() 
        us_perm = us[elems_perm]
        D_broadcasted = D[..., lib.newaxis, :, :] #Add permutation dimension

        us_result = self.tsitsiklis_update_line(xs_perm[..., 0, :], xs_perm[..., 1, :],
                                D_broadcasted, us_perm[..., 0], lib=lib)
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new

    def calculate_all_triang_updates(self, elems_perm, xs_perm, D, us, lib=np):
        """Calculates all triangle updates for all element permutations and computes their minimum as the new solution of :math:`\\phi`.
            For more details on the parameters and return values, see :meth:`calculate_all_line_updates`.
        """
        us_new = us.copy()

        us_perm = us[elems_perm]
        D_broadcasted = D[..., lib.newaxis, :, :] #Add permutation dimension

        us_result = self.tsitsiklis_update_triang(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :],
                                                  D_broadcasted, us_perm[..., 0], us_perm[..., 1], lib=lib)

        #Now we need to take the minimum result of old and all new
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new

    def calculate_all_tetra_updates(self, elems_perm, xs_perm, D, us, lib=np):
        """Calculates all tetrahedral updates for all element permutations and computes their minimum as the new solution of :math:`\\phi`.
            For more details on the parameters and return values, see :meth:`calculate_all_line_updates`.
        """
        us_new = us.copy() 
        us_perm = us[elems_perm]
        D_broadcasted = D[..., lib.newaxis, :, :] #Add permutation dimension

        us_result = self.tsitsiklis_update_tetra(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :], xs_perm[..., 3, :],
                                D_broadcasted, us_perm[..., 0], us_perm[..., 1], us_perm[..., 2], lib=lib)
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new

    def calculate_specific_tetra_updates(self, elems_perm, xs_perm, D, us, lib=np):
        us_new = us.copy() 
        us_perm = us[elems_perm]
        us_result = self.tsitsiklis_update_tetra(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :], xs_perm[..., 3, :],
                                D, us_perm[..., 0], us_perm[..., 1], us_perm[..., 2], lib=lib)
        lib.minimum.at(us_new, elems_perm[..., -1], us_result)

        return us_new


    @abstractmethod
    def _comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
        """Internal call to the concrete implementation of the FIM (see :meth:`comp_fim` for the parameter description).
        """
        ...  # pragma: no cover

    def comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
        """Computes the solution :math:`\\phi` to the anisotropic eikonal equation

        .. math::
            \\left\\{
            \\begin{array}{rll}
            \\left<\\nabla \\phi, D \\nabla \\phi \\right> &= 1 \\quad &\\text{on} \\; \\Omega \\\\
            \\phi(\\mathbf{x}_0) &= g(\\mathbf{x}_0) \\quad &\\text{on} \\; \\Gamma
            \\end{array}
            \\right. .

        Parameters
        ----------
        x0 : ndarray (int)
            Array of [k] discrete point indices of the mesh where we prescribe initial values :math:`\\mathbf{x}_0`.
        x0_vals : ndarray (float)
            Array of [k] discrete prescribed initial values that prescribe :math:`g(\\mathbf{x}_0)`.
        metrics : np.ndarray(float), optional
            Specifies the tensor :math:`D` of the anisotropic eikonal equation as a discrete [m, d, d] array. 
            This is optional **only** if you specified the metrics already at construction time (:class:`FIMBase`), by default None
        max_iterations : int, optional
            Maximum number of iterations before aborting the algorithm.
            If the algorithm stops before reaching convergence, some vertices might still be set to :attr:`undef_val`.
            By default int(1e10)

        Returns
        -------
        ndarray (float, cupy or numpy)
            The solution to the anisotropic eikonal equation, :math:`\\phi` as a [n] array.
        """
        #Suppress warnings of the computations, since they should be handled internally
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            #TODO: Maybe use identity in this case?
            assert metrics is not None or self.metrics is not None, f"Metrics (D) need to be provided in comp_fim, or at construction in __init__"
            if metrics is not None:
                self.check_metrics_argument(metrics)
                metrics = np.linalg.inv(metrics).astype(self.precision) #The inverse metric is used in the FIM algorithm

            return self._comp_fim(x0, x0_vals, metrics, max_iterations=max_iterations)
