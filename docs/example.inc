.. code-block:: python

    import numpy as np
    import cupy as cp
    from fimpy.solver import FIMPY
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt

    #Create triangulated points in 2D
    x = np.linspace(-1, 1, num=50)
    X, Y = np.meshgrid(x, x)
    points = np.stack([X, Y], axis=-1).reshape([-1, 2]).astype(np.float32)
    elems = Delaunay(points).simplices
    elem_centers = np.mean(points[elems], axis=1)

    #The domain will have a small spot where movement will be slow
    velocity_f = lambda x: (1 / (1 + np.exp(3.5 - 25*np.linalg.norm(x - np.array([[0.33, 0.33]]), axis=-1)**2)))
    velocity_p = velocity_f(points) #For plotting
    velocity_e = velocity_f(elem_centers) #For computing
    D = np.eye(2, dtype=np.float32)[np.newaxis] * velocity_e[..., np.newaxis, np.newaxis] #Isotropic propagation

    x0 = np.array([np.argmin(np.linalg.norm(points, axis=-1), axis=0)])
    x0_vals = np.array([0.])

    #Create a FIM solver, by default the GPU solver will be called with the active list
    #Set device='cpu' to run on cpu and use_active_list=false to use Jacobi method
    fim = FIMPY.create_fim_solver(points, elems, D)
    phi = fim.comp_fim(x0, x0_vals)

    #Plot the data of all points to the given x0 at the center of the domain
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    cont_f1 = axes[0].contourf(X, Y, phi.get().reshape(X.shape))
    axes[0].set_title("Distance from center")

    cont_f2 = axes[1].contourf(X, Y, velocity_p.reshape(X.shape))
    axes[1].set_title("Assumed isotropic velocity")
    plt.show()
