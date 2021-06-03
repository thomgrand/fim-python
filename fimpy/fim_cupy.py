"""This file contains the GPU implementation of the Fast Iterative Method, based on cupy.
"""

import numpy as np
import cupy as cp
import cupyx as cpx
from .fim_base import FIMBase
from .utils.comp import metric_norm_matrix_2D_cupy, metric_norm_matrix_3D_cupy, metric_norm_matrix, metric_sqr_norm_matrix_2D_cupy, metric_sqr_norm_matrix_3D_cupy, metric_sqr_norm_matrix
from .cupy_kernels import compute_perm_kernel_str, compute_perm_kernel_shared

@cp.fuse()
def u3_comp_cupy_2D(x1, x2, x3, u1, u2, D):
  """Custom cupy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_triang` to speed up computations for :math:`d = 2`.
  """
  k = u1 - u2
  z2 = x2 - x3
  z1 = x1 - x2
  p11 = metric_sqr_norm_matrix_2D_cupy(D, z1, z1)
  p12 = metric_sqr_norm_matrix_2D_cupy(D, z1, z2)
  p22 = metric_sqr_norm_matrix_2D_cupy(D, z2, z2)
  denominator = p11 - k**2
  sqrt_val = (p11 * p22 - p12**2) / denominator
  sqrt_invalid_mask = sqrt_val < 0.
  sqrt_op = cp.sqrt(sqrt_val)
  rhs = k * sqrt_op
  alpha1 = -(p12 + rhs) / p11
  alpha2 = -(p12 - rhs) / p11
  alpha1 = cp.minimum(cp.maximum(alpha1, 0.), 1.)
  alpha2 = cp.minimum(cp.maximum(alpha2, 0.), 1.)

  u3 = []
  for alpha in [alpha1, alpha2]:
    x = x3 - (alpha[..., cp.newaxis] * x1 + (1 - alpha[..., cp.newaxis]) * x2)
    u3.append(alpha * u1 + (1 - alpha) * u2
              + metric_norm_matrix_2D_cupy(D, x, x))

  return cp.minimum(*u3), sqrt_invalid_mask

#@cp.fuse()
def u3_comp_cupy_3D(x1, x2, x3, u1, u2, D):
  """Custom cupy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_triang` to speed up computations for :math:`d = 3`.
  """
  k = u1 - u2
  z2 = x2 - x3
  z1 = x1 - x2
  p11 = metric_sqr_norm_matrix_3D_cupy(D, z1, z1)
  p12 = metric_sqr_norm_matrix_3D_cupy(D, z1, z2)
  p22 = metric_sqr_norm_matrix_3D_cupy(D, z2, z2)
  denominator = p11 - k**2
  sqrt_val = (p11 * p22 - p12**2) / denominator
  sqrt_invalid_mask = sqrt_val < 0.
  sqrt_op = cp.sqrt(sqrt_val)
  rhs = k * sqrt_op
  alpha1 = -(p12 + rhs) / p11
  alpha2 = -(p12 - rhs) / p11
  alpha1 = cp.minimum(cp.maximum(alpha1, 0.), 1.)
  alpha2 = cp.minimum(cp.maximum(alpha2, 0.), 1.)

  u3 = []
  for alpha in [alpha1, alpha2]:
    x = x3 - (alpha[..., cp.newaxis] * x1 + (1 - alpha[..., cp.newaxis]) * x2)
    u3.append(alpha * u1 + (1 - alpha) * u2
              + metric_norm_matrix_3D_cupy(D, x, x))

  return cp.minimum(*u3), sqrt_invalid_mask


#@cp.fuse() #TODO: formal parameter space overflow 
def u3_comp_cupy_ND(x1, x2, x3, u1, u2, D):
  """Custom cupy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_triang` to speed up computations for :math:`d \notin \{2, 3\}`.
  """
  #p11, p12, sqrt_invalid_mask, rhs = u3_comp_cupy_ND_part1(x1, x2, x3, u1, u2, D)
  #return u3_comp_cupy_ND_part2(p11, p12, rhs, x1, x2, x3, u1, u2, D), sqrt_invalid_mask
  k = u1 - u2
  z2 = x2 - x3
  z1 = x1 - x2
  p11 = metric_sqr_norm_matrix(D, z1, z1, cp)
  p12 = metric_sqr_norm_matrix(D, z1, z2, cp)
  p22 = metric_sqr_norm_matrix(D, z2, z2, cp)
  denominator = p11 - k**2
  sqrt_val = (p11 * p22 - p12**2) / denominator
  sqrt_invalid_mask = sqrt_val < 0.
  sqrt_op = cp.sqrt(sqrt_val)
  rhs = k * sqrt_op
  alpha1 = -(p12 + rhs) / p11
  alpha2 = -(p12 - rhs) / p11
  alpha1 = cp.minimum(cp.maximum(alpha1, 0.), 1.)
  alpha2 = cp.minimum(cp.maximum(alpha2, 0.), 1.)

  u3 = []
  for alpha in [alpha1, alpha2]:
    x = x3 - (alpha[..., cp.newaxis] * x1 + (1 - alpha[..., cp.newaxis]) * x2)
    u3.append(alpha * u1 + (1 - alpha) * u2
              + metric_norm_matrix(D, x, x, cp))

  return cp.minimum(*u3), sqrt_invalid_mask

@cp.fuse()
def tsitsiklis_update_tetra_quadr_cupy_3D(D, k, z1, z2):
  """Custom cupy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_tetra_quadr` to speed up computations for :math:`d = 3`.
  """
  norm_sqr_f = metric_sqr_norm_matrix_3D_cupy
  p11 = norm_sqr_f(D, z1, z1)
  p12 = norm_sqr_f(D, z1, z2)
  p22 = norm_sqr_f(D, z2, z2)
  denominator = p11 - k*k
  sqrt_val = (p11 * p22 - (p12 * p12)) / denominator
  rhs = k * cp.sqrt(sqrt_val)
  alpha1 = -(p12 + rhs) / p11
  #alpha2 = -(p12 - rhs) / p11

  return alpha1 #, alpha2

class FIMCupy(FIMBase):
  """This class implements the Fast Iterative Method on the GPU using cupy.
    The employed algorithm is the Jacobi algorithm, updating all nodes in each iteration.
    For details on the parameters, see :class:`fimpy.fim_base.FIMBase`.
  """
  def __init__(self, points, elems, metrics=None, precision=np.float32):
    super(FIMCupy, self).__init__(points, elems, metrics, precision, comp_connectivities=False)
    #Convert
    #self.nh_map = cp.array(self.nh_map)
    #self.point_elem_map = cp.array(self.point_elem_map)
    self.phi_sol = cp.array(self.phi_sol)
    self.points_perm = cp.array(self.points_perm)
    self.elems_perm = cp.array(self.elems_perm)

    if self.metrics is not None:
      self.metrics = cp.array(self.metrics)
    
    self.streams = [cp.cuda.Stream(non_blocking=False) for i in range(4)]
    self.mempool = cp.get_default_memory_pool()

  def __del__(self):
    #self.mempool.free_all_blocks()
    pass

  def calculate_all_line_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy() 
    us_perm = us[elems_perm]
    D_broadcasted = D[..., cp.newaxis, :, :] #Add permutation dimension

    us_result = self.tsitsiklis_update_line(xs_perm[..., 0, :], xs_perm[..., 1, :],
                            D_broadcasted, us_perm[..., 0], lib=cp)
    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)

    return us_new

  def calculate_all_triang_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy()
    us_perm = us[elems_perm]
    D_broadcasted = D[..., lib.newaxis, :, :] 

    us_result = self.tsitsiklis_update_triang(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :],
                                              D_broadcasted, us_perm[..., 0], us_perm[..., 1], lib=lib)

    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)
    return us_new

  def tsitsiklis_update_tetra(self, x1, x2, x3, x4, D, u1, u2, u3, lib=np):

    with self.streams[0]:
      u_tet = self.calculate_tet_update(x1, x2, x3, x4, D, u1, u2, u3, cp)
      u_tet = lib.where(cp.isnan(u_tet), lib.inf, u_tet)

    #Face calculations (Includes possible line calculations)
    with self.streams[1]:
      triang1 = self.tsitsiklis_update_triang(x1, x2, x4, D, u1, u2, cp)
    with self.streams[2]:
      triang2 = self.tsitsiklis_update_triang(x1, x3, x4, D, u1, u3, cp)
    with self.streams[3]:
      triang3 = self.tsitsiklis_update_triang(x2, x3, x4, D, u2, u3, cp)

    u_triang = lib.minimum(triang1, triang2)
    u_triang = lib.minimum(u_triang, triang3)

    return lib.minimum(u_triang, u_tet)

  def calculate_all_tetra_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy() 
    us_perm = us[elems_perm]
    D_broadcasted = D[..., cp.newaxis, :, :] #Add permutation dimension

    us_result = self.tsitsiklis_update_tetra(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :], xs_perm[..., 3, :],
                            D_broadcasted, us_perm[..., 0], us_perm[..., 1], us_perm[..., 2], lib=cp)
    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)

    return us_new


  def _comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
    if metrics is None:
      metrics = self.metrics

    D = metrics
    self.phi_sol[:] = self.undef_val
    self.phi_sol[x0] = x0_vals

    if type(D) != cp.ndarray:
      D = cp.array(D)

    if D.dtype != self.precision:
      D = D.astype(self.precision)

    self.total_points_calculated = 0 #TODO: Just for testing the performance for PIEMAP

    for i in range(max_iterations):
      self.total_points_calculated += self.elems_perm[..., -1].size
      u_new = self.update_all_points(self.elems_perm, self.points_perm, D, self.phi_sol,
                                        lib=cp)

      if cp.allclose(u_new, self.phi_sol):
        break

      self.phi_sol = u_new

    self.phi_sol = u_new

    return self.phi_sol.copy()

  
  def tsitsiklis_update_tetra_quadr(self, D, k, z1, z2, lib=np):
    if self.dims == 3:
      return tsitsiklis_update_tetra_quadr_cupy_3D(D, k, z1, z2)
    else:
      return super().tsitsiklis_update_tetra_quadr(D, k, z1, z2, cp)


class FIMCupyAL(FIMBase):
  """This class implements the Fast Iterative Method on the GPU using cupy.
    The employed algorithm is the active list algorithm (as proposed in the original paper), updating only a current estimation of the wavefront.
    For details on the parameters, see :class:`fimpy.fim_base.FIMBase`.
  """

  def __init__(self, points, elems, metrics=None, precision=np.float32):
    super(FIMCupyAL, self).__init__(points, elems, metrics, precision, comp_connectivities=True)
    #Convert
    #self.build_linear_nh_elem_map()
    self.nh_map = cp.array(self.nh_map)
    self.point_elem_map = cp.array(self.point_elem_map)
    self.phi_sol = cp.array(self.phi_sol)
    self.points_perm = cp.array(self.points_perm)
    self.elems_perm = cp.array(self.elems_perm)

    if self.metrics is not None:
      self.metrics = cp.array(self.metrics)

    self.active_list = cp.array(self.active_list)
    self.mempool = cp.get_default_memory_pool()
    #self.mempool.set_limit(size=self.nr_elems * 4 * 8 * 250)
    self.norm = self.norm_map[cp][self.dims]

    if self.dims == 2:
      self.u3_comp_cupy = u3_comp_cupy_2D
    elif self.dims == 3:
      self.u3_comp_cupy = u3_comp_cupy_3D
    else:
      self.u3_comp_cupy = u3_comp_cupy_ND

    self.parallel_blocks_perm_kernel = 128
    #self.block_dims = (1024,)
    #self.perm_kernel = cp.RawKernel(compute_perm_kernel_str.replace("{active_perms}", str(self.elem_dims)).replace("{parallel_blocks}", str(self.parallel_blocks_perm_kernel)), 
    #                                'perm_kernel') #, backend='nvcc') #, options=("--device-c",))

    #self.block_dims = (64, 16,1) #Old
    self.block_dims = (1024, 1,1) 
    self.perm_kernel = cp.RawKernel(compute_perm_kernel_shared.replace("{active_perms}", str(self.elem_dims)).replace("{parallel_blocks}", str(self.parallel_blocks_perm_kernel)).replace("{shared_buf_size}", str(4096)), 
                                    'perm_kernel', options=('-std=c++11',)) #, backend='nvcc') #, options=("--device-c",))
    self.streams = [cp.cuda.Stream(non_blocking=False) for i in range(7)]

    #To replace cp.unique
    self.elem_unique_map = cp.zeros(shape=[self.nr_elems], dtype=cp.int32)
    self.elem_ones = cp.ones(shape=[self.nr_elems], dtype=cp.int32)
    self.points_unique_map = cp.zeros(shape=[self.nr_points], dtype=cp.int32)
    self.points_ones = cp.ones(shape=[self.nr_points], dtype=cp.int32)

  def comp_unique_map(self, inds, elem_map=True):
    """Efficient implementation to compute an unique list of indices of active elements/points

    Parameters
    ----------
    inds : ndarray (int)
        An [?] array containing the non-unique active element/point indices
    elem_map : bool, optional
        Marks if the unique indices you want to compute are point or element indices, by default True

    Returns
    -------
    ndarray (int)
        An [?] array holding the indices of the unique element/point indices.
    """
    if elem_map:
      cpx.scatter_add(self.elem_unique_map, inds, self.elem_ones[inds])
      #self.elem_unique_map[inds] = 1
      unique_inds = self.elem_unique_map.nonzero()[0]
      self.elem_unique_map[:] = 0 #Reset for next run
    else:
      cpx.scatter_add(self.points_unique_map, inds, self.points_ones[inds])
      #self.points_unique_map[inds] = 1
      unique_inds = self.points_unique_map.nonzero()[0]
      self.points_unique_map[:] = 0 #Reset for next run

    return unique_inds

  def __del__(self):
    self.mempool.free_all_blocks()
    #pass
  
  def tsitsiklis_update_triang(self, x1, x2, x3, D, u1, u2, lib=np, use_streams=None):
        #norm_f, norm_sqr_f = self.norm_map[lib][D.shape[-1]]

        #Called for non tetra meshes -> Streams are available
        if self.elem_dims == 3:
          assert(use_streams is None)
          with self.streams[0]: 
            u3, sqrt_invalid_mask = self.u3_comp_cupy(x1, x2, x3, u1, u2, D)
            
          with self.streams[1]:
            u3_point = self.tsitsiklis_update_point_sol(x1, x2, x3, D, u1, u2, lib=lib)

        #Use specific streams given by the calling function
        else:
          assert(use_streams is not None and len(use_streams) == 2)
          with use_streams[0]:
            u3, sqrt_invalid_mask = self.u3_comp_cupy(x1, x2, x3, u1, u2, D)
          with use_streams[1]:
            u3_point = self.tsitsiklis_update_point_sol(x1, x2, x3, D, u1, u2, lib=lib)

        #u3_point = u3_point_sol_cupy_2D(x1, x2, x3, D, u1, u2)
        u3_computed = lib.where(sqrt_invalid_mask, u3_point, u3)
        u3_final = u3_computed

        return u3_final

  def calculate_specific_triang_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy()

    us_perm = us[elems_perm]

    us_result = self.tsitsiklis_update_triang(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :],
                                              D, us_perm[..., 0], us_perm[..., 1], lib=lib)

    #Now we need to take the minimum result of old and all new
    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)

    return us_new

  def calculate_specific_line_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy() 
    us_perm = us[elems_perm]
    us_result = self.tsitsiklis_update_line(xs_perm[..., 0, :], xs_perm[..., 1, :],
                            D, us_perm[..., 0], lib=lib)
    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)

    return us_new
  
  def tsitsiklis_update_tetra(self, x1, x2, x3, x4, D, u1, u2, u3, lib=np):

    with self.streams[0]:
      u_tet = self.calculate_tet_update(x1, x2, x3, x4, D, u1, u2, u3, cp)
      u_tet = lib.where(cp.isnan(u_tet), lib.inf, u_tet)

    #Face calculations (Includes possible line calculations)
    #with self.streams[1]:
    triang1 = self.tsitsiklis_update_triang(x1, x2, x4, D, u1, u2, cp, (self.streams[1], self.streams[2]))
    #with self.streams[2]:
    triang2 = self.tsitsiklis_update_triang(x1, x3, x4, D, u1, u3, cp, (self.streams[3], self.streams[4]))
    #with self.streams[3]:
    triang3 = self.tsitsiklis_update_triang(x2, x3, x4, D, u2, u3, cp, (self.streams[5], self.streams[6]))

    u_triang = lib.minimum(triang1, triang2)
    u_triang = lib.minimum(u_triang, triang3)

    return lib.minimum(u_triang, u_tet)

  def tsitsiklis_update_tetra_quadr(self, D, k, z1, z2, lib=np):
    if self.dims == 3:
      return tsitsiklis_update_tetra_quadr_cupy_3D(D, k, z1, z2)
    else:
      return super().tsitsiklis_update_tetra_quadr(D, k, z1, z2, cp)

  #def tsitsiklis_update_tetra(self, x1, x2, x3, x4, D, u1, u2, u3, lib=np):



  def calculate_specific_tetra_updates(self, elems_perm, xs_perm, D, us, lib=np):
    us_new = us.copy() 
    us_perm = us[elems_perm]
    us_result = self.tsitsiklis_update_tetra(xs_perm[..., 0, :], xs_perm[..., 1, :], xs_perm[..., 2, :], xs_perm[..., 3, :],
                            D, us_perm[..., 0], us_perm[..., 1], us_perm[..., 2], lib=cp)
    cpx.scatter_min(us_new, elems_perm[..., -1], us_result)

    return us_new

  def comp_marked_points(self, phi, D, active_inds, use_buffered_vals=False):
    """Computes the update of only the desired points.

    Parameters
    ----------
    phi : np.ndarray (precision)
        [N] array of the values for :math:`\\phi_i` to be used for the update.
    D : np.ndarray (precision)
        [M, d, d] array of :math:`M` to be used for the updates.
    active_inds : np.ndarray (int)
        [?] array of indices that will be updated

    Returns
    -------
    np.ndarray (precision)
        [N] array holding the updated values :math:`\\phi_{i+1}` where only indices found in ``active_inds`` were updated.
    """
    if active_inds.size == 0:
      return phi

    if not use_buffered_vals:
      #"""
      #active_elem_mask = np.any(self.elems[:, np.newaxis, :] == active_inds[np.newaxis, :, np.newaxis], axis=(-1, -2))
      #active_elem_inds = active_elem_mask.nonzero()[0]
      #tmp = cp.sort(self.point_elem_map[active_inds].reshape([-1])) #Performance test -> Future possible implementation
      #active_elem_inds = cp.unique(self.point_elem_map[active_inds]) #Compute only what's necessary #cp.unique(self.point_elem_map[active_inds].reshape([-1]))
      active_elem_inds = self.comp_unique_map(self.point_elem_map[active_inds].reshape([-1]), elem_map=True)
      #active_elem_inds = self.point_elem_map[active_inds].reshape([-1]) #Redundant computations
      active_elems_perm = self.elems_perm[active_elem_inds]

      #Custom kernel to avoid the huge memory overhead for comparisons
      perm_mask_output = cp.zeros(shape=active_elems_perm.shape[0:2], dtype=bool)
      #self.perm_kernel((1,), (1,), (cp.ascontiguousarray(active_elems_perm[..., -1].T), active_inds.astype(cp.int32), perm_mask_output, active_elems_perm.shape[0], active_inds.size))
      self.perm_kernel((self.parallel_blocks_perm_kernel,), self.block_dims, (cp.ascontiguousarray(active_elems_perm[..., -1]), active_inds.astype(cp.int32), perm_mask_output, active_elems_perm.shape[0], active_inds.size))
      #perm_mask_output = perm_mask_output.T

      #For testing
      #perm_mask = cp.any(active_elems_perm[..., -1, np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1) 
      #assert(cp.all(perm_mask == perm_mask_output))

      perm_inds = perm_mask_output.nonzero()
      #perm_cnts = np.sum(perm_mask, axis=-1)
      #perm_cumsum = np.concatenate([[0], np.cumsum(perm_cnts)])
      #assert (cp.all(cp.sum(perm_mask, axis=-1) > 0))

      self.active_elems_perm = active_elems_perm[perm_inds] # = self.elems_perm[active_elem_inds][perm_mask]
      self.active_points_perm = self.points_perm[active_elem_inds][perm_inds]
      #active_D = D[active_elem_inds]
      #active_D = D[active_elems_perm]
      self.active_D = D[active_elem_inds][perm_inds[0]] #cp.tile(D[active_elem_inds, np.newaxis], [1, active_elems_perm.shape[1], 1, 1])[perm_inds]
      #"""
      #active_elem_inds = cp.unique(self.point_elem_map[active_inds])
      #self.active_elems_perm, self.active_points_perm, self.active_D = tmp(self.elems_perm, self.point_elem_map, self.points_perm, D, active_inds, active_elem_inds)
      #active_elem_inds, active_elems_perm, active_points_perm, perm_inds, active_D = comp_marked_points_fuse(active_inds, self.point_elem_map, self.elems_perm)
      #active_elem_inds, active_elems_perm, perm_inds = comp_marked_points_njit(active_inds, self.point_elem_map, self.elems_perm)
      #active_elems_perm = active_elems_perm[perm_inds]
      #active_points_perm = self.points_perm[active_elem_inds][perm_inds]
      #active_D = np.tile(D[active_elem_inds, np.newaxis], [1, active_elems_perm.shape[1], 1, 1])[perm_inds]

    self.total_points_calculated += self.active_elems_perm[..., -1].size
    u_new = self.update_specific_points(self.active_elems_perm,  # self.elems_perm,
                                    self.active_points_perm, #self.points_perm,
                                    self.active_D,
                                    phi,
                                    lib=cp)

    return u_new


  def _comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
    if metrics is None:
      metrics = self.metrics

    D = metrics
    self.phi_sol[:] = self.undef_val
    self.phi_sol[x0] = x0_vals

    if type(D) != cp.ndarray:
      D = cp.array(D)

    if D.dtype != self.precision:
      D = D.astype(self.precision)

    self.active_list[:] = False
    self.active_list[cp.unique(self.nh_map[x0])] = True
    update_al_interval = 1
    active_inds = self.active_list.nonzero()[0]
    self.total_points_calculated = 0 #TODO: Just for testing the performance for PIEMAP
    for i in range(max_iterations):
      #active_points = self.points[self.active_list]
      
      u_new = self.comp_marked_points(self.phi_sol, D, active_inds, use_buffered_vals=(i != 0))

      if i % update_al_interval == 0:
        converged = ((cp.abs(u_new - self.phi_sol) < self.convergence_eps) & (self.active_list))
        converged_inds = converged.nonzero()[0]
        if converged_inds.size > 0:
          #converged_neighbors = cp.unique(self.nh_map[converged_inds])
          converged_neighbors = self.comp_unique_map(self.nh_map[converged_inds].reshape([-1]), elem_map=False)
          u_neighs = self.comp_marked_points(u_new, D, converged_neighbors)
          neighbors_needing_updates = converged_neighbors[((cp.abs(u_new[converged_neighbors] - u_neighs[converged_neighbors]) >= self.convergence_eps))]
          #Check if the neighbors converged
          self.active_list[neighbors_needing_updates] = True #Add neighbors to the active list
          self.active_list[converged_inds] = False #Remove converged points from the active list
          active_inds = self.active_list.nonzero()[0]

          #Use the newly computed values in the next iteration, since they are strictly smaller
          u_new = u_neighs

        if active_inds.size == 0: #cp.all(~self.active_list): #np.allclose(u_new, self.phi_sol, atol=self.convergence_eps):
          break

      self.phi_sol = u_new

    self.phi_sol = u_new

    return self.phi_sol.copy()
