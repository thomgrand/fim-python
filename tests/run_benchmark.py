import os
import sys
os.chdir("/home/thomas/aniso_diff/fim_python/")
sys.path.append("/home/thomas/aniso_diff/fim_python/")
import numpy as np
import cupy as cp
import scipy.io as sio
from fimpy.solver import FIMPY
import time
import timeit
from IPython import get_ipython
import json

from generate_test_data import bench_dims, test_elem_dims, bench_resolutions, elem_fnames, sanity_size_check

def run_single_test(device, use_active_list, dims, elem_dims, resolution, bench_dict=None):
    data_dir = os.path.join(os.path.dirname(__file__), "benchmark_data")
    ipython = get_ipython()

    elem_fname = elem_fnames[elem_dims]
    fname = "elem_dims_%d_dims_%d_resolution_%d_%s.mat" % (elem_dims, dims, resolution, elem_fname)
    fname = os.path.join(data_dir, fname)
    assert(os.path.isfile(fname)) #If you fail here, the generation of test data using generate_test_data.py failed or was not performed

    data = sio.loadmat(fname)
    points, elems, D = data["points"], data["elems"], data["D"]
    nr_points = points.shape[0]

    center_x0 = np.array([np.argmin(np.sum((points - np.mean(points, axis=0, keepdims=True))**2, 
                            axis=-1))])
    #x0_vals = np.array([0.])

    #Setup the solver
    bt = time.time()
    solver = FIMPY.create_fim_solver(points, elems, D, precision=np.float32, device=device, use_active_list=use_active_list)
    at = time.time()
    setup_time = at - bt

    x0 = center_x0
    x0_vals = np.array([0.])

    #Dry run for compilation
    bt = time.time()
    phi1 = solver.comp_fim(x0, x0_vals)
    at = time.time()
    init_run_time = at - bt
    #return #For profiling

    #
    def eval_fim():
        result = solver.comp_fim(x0, x0_vals)
        if device == 'gpu':
            cp.cuda.runtime.deviceSynchronize()
        return result

    if ipython is None:
        timing_avg = timeit.timeit(lambda: eval_fim(), number=5) / 5
        print("Setup time: %f, Avg. compute time: %f, First run time: %f" % (setup_time, timing_avg, init_run_time))
    else:
        timing = ipython.run_line_magic("timeit", "-o eval_fim()")
        timing_avg = timing.average

    if bench_dict is not None:
        bench_dict["device"].append(device)
        bench_dict["active_list"].append(use_active_list)
        bench_dict["dims"].append(dims)
        bench_dict["elem_dims"].append(elem_dims)
        bench_dict["runtime"].append(timing_avg)
        bench_dict["setup_time"].append(setup_time)
        bench_dict["elem_fname"].append(elem_fname)
        bench_dict["nr_points"].append(points.shape[0])
        bench_dict["nr_elems"].append(elems.shape[0])
        bench_dict["resolution"].append(resolution)

    print("Finished %s on %s, active list: %s" % (fname, device, use_active_list))
    
    return setup_time, timing_avg, solver, fname #, points.shape[0], elems.shape[0]
    
if __name__ == "__main__":
    bench_dict = {"device": [], "active_list": [], "dims": [], "elem_dims": [], "runtime": [], "setup_time": [], "elem_fname": [], "nr_points": [], "nr_elems": [],
                    "resolution": []}

    for device in ['cpu', 'gpu']:
        for use_active_list in [True, False]:
            
            for dims in bench_dims:
                if device == 'cpu' and dims not in [1, 2, 3]: #if device == 'cpu' and not use_active_list and dims not in [1, 2, 3]:
                    continue #TODO: Too slow currently (not using any multithreading)

                if device == 'cpu' and dims > 5:
                    continue

                for elem_dims in test_elem_dims:
                    if (dims < elem_dims - 1):
                        continue

                    #if dims > 5 and elem_dims > 2 and use_active_list: #elem_dims == 4 and use_active_list and dims > 3 and resolution > 10 and device == 'gpu':
                    #    continue #Very slow

                    for resolution in bench_resolutions[elem_dims-1]:
                        #Skip very large examples
                        if sanity_size_check(dims, elem_dims, resolution):
                            continue

                        if elem_dims == 4 and resolution == 100:
                            continue
                        
                        if device == 'cpu' and ((elem_dims == 2 and resolution > 500) or (elem_dims == 4 and resolution > 25) or (elem_dims == 3 and resolution > 250)):
                            continue

                        if device == 'cpu' and dims > 3 and elem_dims == 4 and resolution > 10:
                            continue

                        solver, fname = run_single_test(device, use_active_list, dims, elem_dims, resolution, bench_dict)[-2:]
                        if hasattr(solver, 'mempool'):
                            solver.mempool.free_all_blocks()


    with open(os.path.join(os.path.dirname(__file__), "benchmark_results_w_cpu.json"), "w") as bench_f:
        json.dump(bench_dict, bench_f, indent=2, sort_keys=True)
    #sio.savemat(os.path.join(os.path.dirname(__file__), "benchmark_results.mat"), bench_dict, do_compression=True)
