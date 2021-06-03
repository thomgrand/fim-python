import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import cupy as cp
import sys
sys.path.append(".")
from fimpy import FIMPY
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.io as sio
import pandas
import json

#TODO
def generate_usage_example(save_fig=True):
    out_fname = os.path.join(os.path.dirname(__file__), *["..", "docs", "figs", "usage_example.jpg"])
    #Create triangulated points in 2D
    x = np.linspace(-1, 1, num=50)
    X, Y = np.meshgrid(x, x)
    points = np.stack([X, Y], axis=-1).reshape([-1, 2]).astype(np.float32)
    elems = Delaunay(points).simplices
    elem_centers = np.mean(points[elems], axis=1)

    #The domain will have a small spot where movement will be slow
    velocity_p = (1 / (1 + np.exp(3.5 - 25*np.linalg.norm(points - np.array([[0.33, 0.33]]), axis=-1)**2)))
    velocity_e = (1 / (1 + np.exp(3.5 - 25*np.linalg.norm(elem_centers - np.array([[0.33, 0.33]]), axis=-1)**2)))
    D = np.eye(2, dtype=np.float32)[np.newaxis] * velocity_e[..., np.newaxis, np.newaxis] #Isotropic propagation

    x0 = np.array([np.argmin(np.linalg.norm(points, axis=-1), axis=0)])
    x0_vals = np.array([0.])

    #Create a FIM solver, by default the GPU solver will be called with the active list
    fim = FIMPY.create_fim_solver(points, elems, D)
    phi = fim.comp_fim(x0, x0_vals)

    #Plot the data of all points to the given x0 at the center of the domain
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    cont_f1 = axes[0].contourf(X, Y, phi.get().reshape(X.shape))
    axes[0].set_title("Distance from center $\\phi(\\mathbf{x})$")
    scatter_h = axes[0].scatter(points[x0, 0], points[x0, 1], marker='x', color='r')
    #cbar = plt.colorbar(cont_f1)
    #cbar.set_label("$\\phi$")
    axes[0].legend([scatter_h], ["$\\mathbf{x}_0$"])

    cont_f2 = axes[1].contourf(X, Y, velocity_p.reshape(X.shape))
    axes[1].set_title("Assumed isotropic velocity  $D(\\mathbf{x}) = c(\\mathbf{x}) I$")
    #cbar = plt.colorbar(cont_f2)
    #cbar.set_label("Velocity")

    fig.set_size_inches((10, 6))
    plt.tight_layout()
    if save_fig:
        fig.savefig(out_fname)
    else:
        plt.show()
        
    plt.close(fig)

def generate_benchmark_plot(benchmark_data_fname, save_fig=True):
    with open(benchmark_data_fname, "r") as bench_f:
        benchmark_data = json.load(bench_f)
    out_fname = os.path.join(os.path.dirname(__file__), *["..", "docs", "figs", "benchmark"])
    all_pd_data = pandas.DataFrame(benchmark_data)

    all_dims = np.unique(all_pd_data["dims"])
    colors = plt.cm.brg(np.linspace(0, 1, num=all_dims.size))

    figs = []
    #GPU
    for device in ["gpu", "cpu"]:
        plt.gca().set_prop_cycle(None) #Reset color cycle
        pd_data = all_pd_data[all_pd_data["device"] == device]
        #colors = plt.cm.get_cmap('jet', all_dims.size)
        fig, axes = plt.subplots(nrows=1, ncols=3)
        lines_hs = {2: [], 3: [], 4: []}
        for elem_dim_i, elem_dims in enumerate([4, 3, 2]): #np.unique(pd_data["elem_dims"])):
            for dim_i, dims in enumerate([1, 2, 3, 5]): #enumerate(all_dims): #[1, 2, 3, 5]
                for use_active_list in [False, True]:
                    plt.sca(axes[elem_dim_i])
                    data = pd_data[(pd_data["active_list"] == use_active_list) & (pd_data["dims"] == dims) & (pd_data["elem_dims"] == elem_dims)]

                    if data.size == 0:
                        continue

                    data = data.sort_values("resolution", ascending=False) 
                    h = 2 / data["resolution"]
                    nr_elems = data["nr_elems"]

                    label=("%d" % (dims) if not use_active_list else None) #('w. AL' if use_active_list else 'w/o AL')
                    lines_h = axes[elem_dim_i].plot(nr_elems.values, data["runtime"].values, linestyle=('--' if use_active_list else '-'), color=colors[dim_i], label=label) #color='b' if use_active_list else 'r')

                    if not use_active_list:
                        lines_hs[elem_dims].append((lines_h[0], dims))

            
            if elem_dim_i != 0:
                #axes[elem_dim_i].set_yticklabels([])
                pass
            else:
                plt.ylabel("Runtime [s]")

            #Make a nice title from the data
            plotted_data = pd_data[pd_data["elem_dims"] == elem_dims]["elem_fname"]
            if plotted_data.size > 0:
                axes[elem_dim_i].set_title(plotted_data.iat[0].capitalize().replace("_d", " D"))
                #axes[elem_dim_i].invert_xaxis()
                #plt.xlabel("$h$")
                plt.xlabel("#Elements")
                plt.xscale("log")
                plt.yscale("log")
                plt.grid(True, which='both', axis='both')

        #Search for valid handles
        used_elem_dims = list(lines_hs.keys())
        len_handles = np.array([len(lines_hs[written_dim]) for written_dim in used_elem_dims])

        if np.any(len_handles > 0):
            handles, plot_dims = zip(*lines_hs[used_elem_dims[int(np.where(len_handles > 0)[0][0])]])        
            axes[1].legend(handles, plot_dims, title="$d=$", loc='center', bbox_to_anchor=(0.15, -0.3, 0.75, 0.25), borderpad=0.8, ncol=2)

        fig.suptitle("Fimpy %s Benchmark" % (device.upper()))
        fig.set_size_inches((14, 8))
        fig.tight_layout()
        figs.append(fig)

        if save_fig:
            fig.savefig(out_fname + "_%s" % (device) + ".jpg")
            fig.savefig(out_fname + "_%s" % (device) + ".pdf")
    plt.show()
    [plt.close(fig) for fig in figs]
    #plt.show()
    #pd_data.

if __name__ == "__main__":
    #generate_usage_example(False)
    generate_benchmark_plot(os.path.join(os.path.dirname(__file__), "benchmark_results_w_cpu.json"), save_fig=True)
