"""This file contains the CPU implementation of the Fast Iterative Method, based on numpy and numba.
"""

from fimpy.utils.comp import metric_norm_matrix_3D_njit
from numba.np.ufunc import parallel
import numpy as np
from .fim_base import FIMBase
#from utils.nh_manager import calculate_all_triang_updates, calculate_specific_triang_updates
from numba import njit
from .fim_cutils import compute_perm_mask

@njit(cache=True, nogil=True)
def tsitsiklis_update_triang_3D(x1, x2, x3, D, u1, u2, p11, p12, p22):
  """Custom numpy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_triang` to speed up computations for :math:`d = 3`.
  """
  k = u1 - u2
  denominator = p11 - k**2
  sqrt_val = (p11 * p22 - p12**2) / denominator
  sqrt_invalid_mask = sqrt_val < 0.
  sqrt_op = np.sqrt(sqrt_val)
  rhs = k * sqrt_op
  alpha1 = -(p12 + rhs) / p11
  alpha2 = -(p12 - rhs) / p11
  alpha1 = np.minimum(np.maximum(alpha1, 0.), 1.)
  alpha2 = np.minimum(np.maximum(alpha2, 0.), 1.)

  u3 = []
  for alpha in [alpha1, alpha2]:
      x = x3 - (np.expand_dims(alpha, axis=-1) * x1 + (1 - np.expand_dims(alpha, axis=-1)) * x2)
      u3.append(alpha * u1 + (1 - alpha) * u2
              + metric_norm_matrix_3D_njit(D, x, x, ret_sqrt=True))

  return np.minimum(u3[0], u3[1]), sqrt_invalid_mask


class FIMNP(FIMBase):
  """This class implements the Fast Iterative Method on the CPU using a combination of numpy and numba.
    The employed algorithm is the Jacobi algorithm, updating all nodes in each iteration.
    For details on the parameters, see :class:`fimpy.fim_base.FIMBase`.
  """

  def __init__(self, points, elems, metrics=None, precision=np.float32):
    super(FIMNP, self).__init__(points, elems, metrics, precision)

  def _comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
    if metrics is None:
      metrics = self.metrics

    D = metrics
    self.phi_sol = np.ones(shape=[self.nr_points], dtype=self.precision) * self.undef_val
    self.phi_sol[x0] = x0_vals

    if D.dtype != self.precision:
      D = D.astype(self.precision)

    for i in range(max_iterations):
      u_new = self.update_all_points(self.elems_perm, self.points_perm, D, self.phi_sol,
                                        lib=np)

      if np.allclose(u_new, self.phi_sol):
        break

      self.phi_sol = u_new

    self.phi_sol = u_new

    return self.phi_sol

class FIMNPAL(FIMBase):
  """This class implements the Fast Iterative Method on the CPU using a combination of numpy and numba.
    The employed algorithm is the active list algorithm (as proposed in the original paper), updating only a current estimation of the wavefront.
    For details on the parameters, see :class:`fimpy.fim_base.FIMBase`.
  """

  def __init__(self, points, elems, metrics=None, precision=np.float32):
    super(FIMNPAL, self).__init__(points, elems, metrics, precision, comp_connectivities=True)


  def tsitsiklis_update_triang(self, x1, x2, x3, D, u1, u2, lib=np):
    """Custom numpy implementation of :meth:`fimpy.fim_base.FIMBase.tsitsiklis_update_triang` to speed up computations for :math:`d = 3`.
    """

    if self.dims == 3:
      z2 = x2 - x3
      z1 = x1 - x2

      p11 = metric_norm_matrix_3D_njit(D, x1=z1, x2=z1, ret_sqrt=False)
      p12 = metric_norm_matrix_3D_njit(D, x1=z1, x2=z2, ret_sqrt=False)
      p22 = metric_norm_matrix_3D_njit(D, x1=z2, x2=z2, ret_sqrt=False)

      u3, sqrt_invalid_mask = tsitsiklis_update_triang_3D(x1, x2, x3, D, u1, u2, p11, p12, p22)
      u3_point = self.tsitsiklis_update_point_sol(x1, x2, x3, D, u1, u2, lib=lib)
      u3_computed = lib.where(sqrt_invalid_mask, u3_point, u3)
      u3_final = u3_computed

      return u3_final
    else:
      return super(FIMNPAL, self).tsitsiklis_update_triang(x1, x2, x3, D, u1, u2)

  def comp_marked_points(self, phi, D, active_inds):
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

    #active_elem_mask = np.any(self.elems[:, np.newaxis, :] == active_inds[np.newaxis, :, np.newaxis], axis=(-1, -2))
    #active_elem_inds = active_elem_mask.nonzero()[0]
    active_elem_inds = np.unique(self.point_elem_map[active_inds].reshape([-1]))
    active_elems_perm = self.elems_perm[active_elem_inds]
    #perm_mask = np.any(active_elems_perm[..., -1, np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1)
    perm_mask = compute_perm_mask(np.ascontiguousarray(active_elems_perm[..., -1]), active_inds)
    #TODO: Testing
    #assert(np.all(perm_mask == np.any(active_elems_perm[..., -1, np.newaxis] == active_inds[np.newaxis, np.newaxis], axis=-1)))
    perm_inds = perm_mask.nonzero()
    #perm_cnts = np.sum(perm_mask, axis=-1)
    #perm_cumsum = np.concatenate([[0], np.cumsum(perm_cnts)])
    #assert (np.all(np.sum(perm_mask, axis=-1) > 0))

    active_elems_perm = active_elems_perm[perm_inds] # = self.elems_perm[active_elem_inds][perm_mask]
    active_points_perm = self.points_perm[active_elem_inds][perm_inds]
    #active_D = D[active_elem_inds]
    #active_D = D[active_elems_perm]
    active_D = D[active_elem_inds][perm_inds[0]]
    #active_D = np.tile(D[active_elem_inds, np.newaxis], [1, active_elems_perm.shape[1], 1, 1])[perm_inds]

    #active_elem_inds, active_elems_perm, perm_inds = comp_marked_points_njit(active_inds, self.point_elem_map, self.elems_perm)
    #active_elems_perm = active_elems_perm[perm_inds]
    #active_points_perm = self.points_perm[active_elem_inds][perm_inds]
    #active_D = np.tile(D[active_elem_inds, np.newaxis], [1, active_elems_perm.shape[1], 1, 1])[perm_inds]
    u_new = self.update_specific_points(active_elems_perm,  # self.elems_perm,
                                    active_points_perm, #self.points_perm,
                                    active_D,
                                    phi,
                                    lib=np)

    return u_new


  def _comp_fim(self, x0, x0_vals, metrics=None, max_iterations=int(1e10)):
    if metrics is None:
      metrics = self.metrics

    D = metrics
    self.phi_sol = np.ones(shape=[self.nr_points], dtype=self.precision) * self.undef_val
    self.phi_sol[x0] = x0_vals
    self.active_list[:] = False
    self.active_list[np.unique(self.nh_map[x0])] = True

    if D.dtype != self.precision:
      D = D.astype(self.precision)

    for i in range(max_iterations):
      #active_points = self.points[self.active_list]
      active_inds = self.active_list.nonzero()[0].astype(np.int32)
      
      u_new = self.comp_marked_points(self.phi_sol, D, active_inds)
      converged = ((np.abs(u_new - self.phi_sol) < self.convergence_eps) & (self.active_list))
      converged_inds = converged.nonzero()[0]
      converged_neighbors = np.unique(self.nh_map[converged_inds])
      u_neighs = self.comp_marked_points(u_new, D, converged_neighbors)
      neighbors_needing_updates = converged_neighbors[((np.abs(u_new[converged_neighbors] - u_neighs[converged_neighbors]) >= self.convergence_eps))]
      #Check if the neighbors converged
      self.active_list[neighbors_needing_updates] = True #Add neighbors to the active list
      self.active_list[converged_inds] = False #Remove converged points from the active list


      if np.all(~self.active_list): #np.allclose(u_new, self.phi_sol, atol=self.convergence_eps):
        break

      self.phi_sol = u_new

    self.phi_sol = u_new

    return self.phi_sol


if __name__ == "__main__":
  import pyvista as pv
  import vtk
  from IPython import get_ipython
  import timeit
  from scipy.spatial import Delaunay
  from fim_cupy import FIMCupy, FIMCupyAL
  ipython = get_ipython()
  #ug = pv.UnstructuredGrid(#"/home/thomas/TUG/thesis/res/vtks/math_prelim/plane_manifold_flat.vtu")
  #                         "S:/Meine Bibliotheken/TUG/thesis/res/vtks/math_prelim/plane_manifold_flat.vtu")
  #points, triangs, D = ug.points[..., :-1], ug.cells_dict[vtk.VTK_TRIANGLE], ug.cell_arrays["metric_tensors"].reshape([-1, 3, 3])[..., :2, :2]
  dims = 3
  elem_dims = 4
  lines = False
  x = np.linspace(-1, 1, num=25)
  assert(elem_dims <= (dims+1))

  if elem_dims == 2:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    points = np.random.uniform(size=[x.size**2, dims])
    repetitions = np.minimum(points.shape[0]*15, points.shape[0]**2) // points.shape[0]
    rows = np.concatenate((np.arange(points.shape[0]),)*repetitions, axis=-1)
    cols = np.random.choice(points.shape[0], size=points.shape[0]*repetitions)
    while np.unique(cols).size != points.shape[0]:
      cols = np.random.choice(points.shape[0], size=points.shape[0]*repetitions)
    data = np.ones(rows.size)
    connectivity_mat = coo_matrix((data, (rows, cols)))
    connectivity_mat = 0.5 * (connectivity_mat + connectivity_mat.T).tocsc()
    span_tree = minimum_spanning_tree(connectivity_mat)
    span_tree.eliminate_zeros()
    elems = np.stack(span_tree.nonzero(), axis=-1)

  elif elem_dims == 3:
    X, Y = np.meshgrid(x, x) #, indexing='ij')
    points = np.stack([X, Y], axis=-1).reshape([-1, 2])
    elems = Delaunay(points[..., :2]).simplices

  elif elem_dims == 4:
    points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape([-1, 3])
    elems = Delaunay(points).simplices

  if points.shape[-1] < dims:
    points = np.concatenate([points, np.zeros(shape=[points.shape[0], dims - points.shape[1]])], axis=-1)

  print("#Points: %d, #Elems: %d" % (points.shape[0], elems.shape[0]))
  elems_centers = np.mean(points[elems], axis=1)
  D = np.eye(dims)[np.newaxis] * (2.5 + np.sin(elems_centers[..., 0]) + np.cos(elems_centers[..., 1]))[:, np.newaxis, np.newaxis]
  #D = np.tile(np.eye(dims), [elems.shape[0], 1, 1])

  #Finalize all
  points = np.array(points, dtype=points.dtype)
  D = np.array(D, dtype=D.dtype)
  #points, triangs, D = ug.points, ug.cells_dict[vtk.VTK_TRIANGLE], ug.cell_arrays["metric_tensors"].reshape([-1, 3, 3])

  #fim_np = FIMNP(points, elems, D, precision=np.float32)
  #fim_np_al = FIMNPAL(points, elems, D, precision=np.float32)
  fim_cp = FIMCupy(points, elems, D, precision=np.float32)
  fim_cp_al = FIMCupyAL(points, elems, D, precision=np.float32)

  np.random.seed(0)
  x0 = np.array([np.random.choice(points.shape[0])])
  x0_vals = np.array([0.])
  #result = fim_np.comp_fim(x0, x0_vals)
  #result_al = fim_np_al.comp_fim(x0, x0_vals)
  result_cp = fim_cp.comp_fim(x0, x0_vals)
  result_cp_al = fim_cp_al.comp_fim(x0, x0_vals)
  #print(timeit.timeit(lambda: fim_np.comp_fim(x0, x0_vals), number=5) / 5)
  #print(timeit.timeit(lambda: fim_np_al.comp_fim(x0, x0_vals), number=5) / 5)
  print(timeit.timeit(lambda: fim_cp.comp_fim(x0, x0_vals), number=5) / 5)
  print(timeit.timeit(lambda: fim_cp_al.comp_fim(x0, x0_vals), number=5) / 5)
  assert (np.allclose(result_cp, result_cp_al, atol=5e-3))
  #assert(np.allclose(result, result_al, atol=1e-3))
  #assert (np.allclose(result_cp, result_al, atol=5e-3))
  import sys
  sys.exit(0)
  timing_wo_al = ipython.run_line_magic("timeit", "-o fim_np.comp_fim(x0, x0_vals)")
  #timing_w_al = ipython.run_line_magic("timeit", "-o fim_np_al.comp_fim(x0, x0_vals)")
  timing_cp = ipython.run_line_magic("timeit", "-o fim_cp.comp_fim(x0, x0_vals)")
  print(timing_wo_al.average, timing_w_al.average, timing_cp.average)
  from matplotlib import tri as mtri, pyplot as plt

  triang_matplotlib = mtri.Triangulation(points[:, 0], points[:, 1], triangs)
  plt.tricontourf(triang_matplotlib, result)
  plt.show()
  #from
  #for i in range(150):
  #  #visualizeUnstructredGridMeshLegacy(points, triangs, point_data_dict={"act_times": u_current},
  #  #                                   outfile_name="/home/thomas/aniso_diff/python/test/output/test_fim_np.%03d.vtk" % (i))
  #  u_current = calculate_all_triang_updates(fim_np.points, fim_np.elems_perm, fim_np.points[fim_np.elems_perm],
  #                                        D, u_current)
  print("Finished")
