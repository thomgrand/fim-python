"""
Testcases:
- Tube
- Sphere
- Cube
- Simplified Heart
- Network
- Use vtkCutter to get 2D data
- With and without anisotropy

"""
import os
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
import numpy as np
#import pyvista as pv
#import vtk
import scipy.io as sio

test_dims = [1, 2, 3, 5]
bench_dims = test_dims + [10, 20] #, 50] #, 100]

test_elem_dims = [2, 3, 4]

test_resolutions = {1: [10, 25, 50],
                    2: [5, 10],
                    3: [5, 10]}

bench_resolutions = {1: test_resolutions[1] + [100, 200, 400, 800, 1500],
                        2: test_resolutions[2] + [25, 50, 100, 200, 300, 500], #1000],
                        3: test_resolutions[3] + [25, 50]} #, 75, 100]}

elem_fnames = {2: "network", 3: "surface", 4: "tetra_domain"}

isotropic_het_vel_f = lambda x, dims, scales=1: np.eye(dims)[np.newaxis] * (dims + 0.5 + np.sum(np.sin(x * 2*np.pi / scales), axis=-1))[:, np.newaxis, np.newaxis]

sanity_size_check = lambda dims, elem_dims, resolution: dims >= 5 and ((elem_dims == 4 and resolution > 50) or (elem_dims == 4 and dims > 10 and resolution > 25) 
                                                                        or (elem_dims == 3 and resolution > 300) or (elem_dims == 2 and resolution > 400))

def generate_test_data(gen_bench_data=False):

    if gen_bench_data:
        data_dir = os.path.join(os.path.dirname(__file__), "benchmark_data")
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    for elem_dims in test_elem_dims:
        valid_dims = np.array(bench_dims if gen_bench_data else test_dims)
        valid_dims = valid_dims[valid_dims >= (elem_dims - 1)] #Request a proper manifold
        resolutions = (bench_resolutions if gen_bench_data else test_resolutions)            

        for resolution in resolutions[elem_dims-1]:
            x = np.linspace(-1, 1, num=resolution)

            #Create the mesh
            #Line network
            if elem_dims == 2: 
                pass #Will be calculated as a point cloud for each dimensional case       

            #Triangular square surface
            elif elem_dims == 3:
                X, Y = np.meshgrid(x, x) #, indexing='ij')
                points = np.stack([X, Y], axis=-1).reshape([-1, 2])
                elems = Delaunay(points[..., :2]).simplices

            #Tetrahedral cube domain
            elif elem_dims == 4:
                points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape([-1, 3])
                elems = Delaunay(points).simplices
            
            for dims in valid_dims:      
                assert(elem_dims <= (dims+1))

                #Skip very large examples
                if sanity_size_check(dims, elem_dims, resolution):
                    continue

                if elem_dims == 2: 
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
                

                elem_fname = elem_fnames[elem_dims]
                #Fill the zero dimensions
                if points.shape[-1] < dims:
                    points = np.concatenate([points, np.zeros(shape=[points.shape[0], dims - points.shape[1]])], axis=-1)

                elem_centers = np.mean(points[elems], axis=1)
                D = isotropic_het_vel_f(elem_centers, dims)

                fname = "elem_dims_%d_dims_%d_resolution_%d_%s.mat" % (elem_dims, dims, resolution, elem_fname)
                if gen_bench_data:
                    data = {"elems": elems.astype(np.int32), "points": points.astype(np.float32), "D": D.astype(np.float32)}
                else:
                    data = {"elems": elems, "points": points, "D": D}
                sio.savemat(os.path.join(data_dir, fname), data, do_compression=True)
                print("Created %s" % (fname))
                #ug = pv.UnstructuredGrid({vtk_elem_type: elems}, points)
                #ug.cell_arrays["D"] = D
                #ug.save(fname)

        

if __name__ == "__main__":
    generate_test_data()