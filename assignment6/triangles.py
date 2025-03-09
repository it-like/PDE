import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def mygrid(g=0, v=0.2):
    """
    Generate a simple 2D triangular mesh 
    g = 0: Unit square [0,1]x[0,1].
    
    v = approximate spacing (max size) of the triangles.
    
    Returns:
      G: placeholder for 'grid geometry data'
      N: (npoints, 2) array of node coordinates
      T: (ntri, 3) array of triangle vertex indices
      F: placeholder for 'boundary info' or other data
    """
    if g == 0:
        # Determine how many intervals in each direction
        nx = int(1.0 / v) + 1  # e.g. step ~ v in [0,1]
        ny = nx
        
        # Build a regular grid of points
        xvals = np.linspace(0, 1, nx)
        yvals = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(xvals, yvals)
        
        # Flatten into a list of points
        N = np.column_stack([X.ravel(), Y.ravel()])
        
        
        T_list = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Indices of the 4 corners of the cell
                p1 = j*nx + i
                p2 = j*nx + (i+1)
                p3 = (j+1)*nx + i
                p4 = (j+1)*nx + (i+1)
                # Two triangles per square cell
                T_list.append([p1, p2, p4])
                T_list.append([p1, p4, p3])
        T = np.array(T_list, dtype=int)
        
    G = None
    F = None
    
    return G, N, T, F

def showmesh(N, T):
    """
    Plot the triangular mesh given by node array N and triangle index array T.
    N: shape (npoints, 2)
    T: shape (ntri, 3)
    """
    plt.figure(figsize=(5,5))
    # Build a Triangulation object for plotting
    triobj = mtri.Triangulation(N[:,0], N[:,1], T)
    # Plot the mesh
    plt.triplot(triobj, color='green')
    plt.axis('equal')
    plt.title("Mesh Visualization")
    plt.show()


if __name__ == "__main__":

    G, N, T, F = mygrid(g=0, v=0.1)
    showmesh(N, T)

    
