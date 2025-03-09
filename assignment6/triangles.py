import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from chicken_mesh import chicken_mesh
def mygrid(G,w, points):
    '''
    Generates a mesh depending on the given input of G.
    w represents frequency

    Returns:
    N: List of nodes [[x_1,y_1],...,[x_n,y_n]] 

    T: List of triangles from nodes in N

    P: Material constant με for each triangle
    '''    
    tol = 1e-6
    
    if G == 1:
        # Single triangle case
        N = np.array([[0, 0], [1, 0], [0, 1]])
        T = np.array([[0, 1, 2, 1, 1, 1]])  # all three edges are boundary
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        P = np.array([mu_l * eps_l])
        return N, T, P

    # Set domain limits based on G:
    if G == 0 or G == 3:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
    else:
        x_min, x_max = 0.0, 2.0
        y_min, y_max = 0.0, 1.0

    # Create structured grid points.
    x = np.linspace(x_min, x_max, points)
    y = np.linspace(y_min, y_max, points)
    X, Y = np.meshgrid(x, y)
    N = np.vstack((X.flatten(), Y.flatten())).T

    # Determine grid dimensions.
    nx = points
    ny = points
    T_list = []
    
    # Loop over each cell and split into two triangles.
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Node indices (using row-major ordering)
            n1 = j * nx + i
            n2 = n1 + 1
            n3 = n1 + nx
            n4 = n3 + 1
            # Triangle 1: bottom-left, bottom-right, top-left
            tri1 = [n1, n2, n3]
            # Triangle 2: bottom-right, top-right, top-left
            tri2 = [n2, n4, n3]
            # For each triangle, determine boundary flags.
            T_list.append(np.concatenate([tri1, 
                            np.array([edge_on_boundary(N, [n1, n2], x_min, x_max, y_min, y_max, tol),
                                      edge_on_boundary(N, [n2, n3], x_min, x_max, y_min, y_max, tol),
                                      edge_on_boundary(N, [n3, n1], x_min, x_max, y_min, y_max, tol)])]))
            T_list.append(np.concatenate([tri2, 
                            np.array([edge_on_boundary(N, [n2, n4], x_min, x_max, y_min, y_max, tol),
                                      edge_on_boundary(N, [n4, n3], x_min, x_max, y_min, y_max, tol),
                                      edge_on_boundary(N, [n3, n2], x_min, x_max, y_min, y_max, tol)])]))
    T = np.array(T_list, dtype=int)
    
    # Material properties
    mu_l = 4 * np.pi * 1e-7
    eps_l = 8.85e-12
    if G == 3:
        # "Chicken" material properties.
        mu_p = 4 * np.pi * 1e-7
        eps_p = 4.43e-11
    P = []
    for tri in T:
        verts = N[tri[:3]]
        centroid = np.mean(verts, axis=0)
        if G == 3:
            # Define chicken region as an ellipse centered at (0.5,0.5) with radii 0.2 and 0.3.
            if ((centroid[0]-0.5)/0.2)**2 + ((centroid[1]-0.5)/0.3)**2 <= 1:
                P.append(mu_p * eps_p)
            else:
                P.append(mu_l * eps_l)
        else:
            P.append(mu_l * eps_l)
    P = np.array(P)
    
    return N, T, P

def edge_on_boundary(N, edge_nodes, x_min, x_max, y_min, y_max, tol):
    """
    Return 1 if both endpoints of the edge lie on one of the domain boundaries.
    """
    p1 = N[edge_nodes[0]]
    p2 = N[edge_nodes[1]]
    # Check if the edge lies on vertical boundaries:
    if (abs(p1[0]-x_min) < tol and abs(p2[0]-x_min) < tol) or (abs(p1[0]-x_max) < tol and abs(p2[0]-x_max) < tol):
        return 1
    # Check if the edge lies on horizontal boundaries:
    if (abs(p1[1]-y_min) < tol and abs(p2[1]-y_min) < tol) or (abs(p1[1]-y_max) < tol and abs(p2[1]-y_max) < tol):
        return 1
    return 0

def plotmygrid(N, T, P):
    """
    Plot the triangulation.
    Triangles are drawn with light gray edges, and boundary edges (flag==1) are drawn thicker in black.
    If the number of nodes is small (=< 25), the node numbers are displayed.
    """
    plt.figure()
    for tri in T:
        verts = N[tri[:3]]
        poly = plt.Polygon(verts, edgecolor='gray', facecolor='none')
        plt.gca().add_patch(poly)
        # Plot each edge with flag==1 in black, thicker.
        for i in range(3):
            if tri[3+i] == 1:
                j = (i+1) % 3
                edge = N[tri[[i, j]]]
                plt.plot(edge[:, 0], edge[:, 1], 'k-', linewidth=2)
    # Optionally, draw the underlying structured grid lines.
    plt.scatter(N[:, 0], N[:, 1], color='red', s=10)
    if N.shape[0] < 26:
        for i, (x, y) in enumerate(N):
            plt.text(x, y, str(i+1), color='blue', fontsize=8)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mesh')
    plt.show()

def mygridrefinement(N, T, P):
    """
    Refine the grid by splitting each triangle into 4 subtriangles.
    New nodes are added at the midpoints of each edge.
    
    Returns:
       new_N: refined node coordinates.
       new_T: refined triangle connectivity (with updated boundary flags).
       new_P: refined material property array.
    """
    new_nodes = N.tolist()
    node_dict = {}  # key: (min_index, max_index), value: new node index
    new_T = []
    new_P = []
    tol = 1e-6

    def get_midpoint(i, j):
        key = tuple(sorted((i, j)))
        if key in node_dict:
            return node_dict[key]
        else:
            pt = (N[i] + N[j]) / 2
            new_nodes.append(pt.tolist())
            idx = len(new_nodes) - 1
            node_dict[key] = idx
            return idx

    # Loop over each triangle in T.
    for k, tri in enumerate(T):
        indices = tri[:3]
        i1, i2, i3 = indices
        m12 = get_midpoint(i1, i2)
        m23 = get_midpoint(i2, i3)
        m31 = get_midpoint(i3, i1)
        # Define 4 new triangles.
        tri1 = [i1, m12, m31]
        tri2 = [m12, i2, m23]
        tri3 = [m31, m23, i3]
        tri4 = [m12, m23, m31]
        def comp_flags(tri_indices):
            flags = []
            for i in range(3):
                j = (i+1)%3
                p1 = new_nodes[tri_indices[i]]
                p2 = new_nodes[tri_indices[j]]
                flag = 0
                if (abs(p1[0]-min(N[:,0]))<tol and abs(p2[0]-min(N[:,0]))<tol) or \
                   (abs(p1[0]-max(N[:,0]))<tol and abs(p2[0]-max(N[:,0]))<tol) or \
                   (abs(p1[1]-min(N[:,1]))<tol and abs(p2[1]-min(N[:,1]))<tol) or \
                   (abs(p1[1]-max(N[:,1]))<tol and abs(p2[1]-max(N[:,1]))<tol):
                    flag = 1
                flags.append(flag)
            return flags
        new_T.append(np.concatenate([tri1, comp_flags(tri1)]))
        new_T.append(np.concatenate([tri2, comp_flags(tri2)]))
        new_T.append(np.concatenate([tri3, comp_flags(tri3)]))
        new_T.append(np.concatenate([tri4, comp_flags(tri4)]))
        new_P += [P[k]] * 4

    new_N = np.array(new_nodes)
    new_T = np.array(new_T, dtype=int)
    new_P = np.array(new_P)
    return new_N, new_T, new_P

def elementstiffmatrix(t):
    """
    Compute the element stiffness matrix for a triangle.
    
    Parameters:
      t: (3,2) array of triangle vertex coordinates.
    
    Returns:
      Se: 3x3 element stiffness matrix.
      A: area of the triangle.
    """
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    A = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    Se = (np.outer(b, b) + np.outer(c, c)) / (4 * A)
    return Se, A

def elementmassmatrix(A):
    """
    Compute the element mass matrix for a triangle of area A.
    For linear elements, the mass matrix is:
       (A/12)*[[2,1,1],[1,2,1],[1,1,2]]
    """
    Me = (A / 12.0) * np.array([[2, 1, 1],
                                [1, 2, 1],
                                [1, 1, 2]])
    return Me


def FEHelmholtz2D(g, N, T, w, P):
    """
    Solve the Helmholtz equation in 2D: Δu + ω²*(μ*ε)*u = 0 in the domain,
    with Dirichlet boundary condition u = g on the boundary.
    
    Parameters:
      g : function g(x,y) prescribing the Dirichlet condition.
      N : nodes array.
      T : triangle connectivity (first 3 columns are node indices).
      w : frequency (ω).
      P : array of material properties for each triangle.
    
    Returns:
      u : the FE solution at the nodes.
      K_global, M_global : the assembled stiffness and mass matrices.
    """
    n_nodes = N.shape[0]
    K_global = np.zeros((n_nodes, n_nodes))
    M_global = np.zeros((n_nodes, n_nodes))
    
    # Assemble element matrices (using linear finite elements)
    for i, tri in enumerate(T):
        indices = tri[:3]
        coords = N[indices]
        Se, area = elementstiffmatrix(coords)
        Me = elementmassmatrix(area)
        # Scale the mass matrix by the element material property.
        Me = P[i] * Me
        for a in range(3):
            for b in range(3):
                K_global[indices[a], indices[b]] += Se[a, b]
                M_global[indices[a], indices[b]] += Me[a, b]
                
    # Global system: A*u = 0, with A = K - ω²*M.
    A_global = K_global - (w**2) * M_global
    
    # Identify boundary nodes from T using the boundary flags (columns 4-6).
    boundary_nodes = set()
    for tri in T:
        for j in range(3):
            if tri[3+j] == 1:
                boundary_nodes.add(tri[j])
    boundary_nodes = np.array(list(boundary_nodes))
    
    # Prescribe Dirichlet data.
    u_D = np.zeros(n_nodes)
    for i in boundary_nodes:
        u_D[i] = g(N[i, 0], N[i, 1])
    
    # Build the right-hand side vector.
    b_vec = np.zeros(n_nodes)
    
    # Impose Dirichlet BCs by modifying the matrix.
    free_nodes = np.setdiff1d(np.arange(n_nodes), boundary_nodes)
    for i in boundary_nodes:
        A_global[i, :] = 0
        A_global[i, i] = 1
        b_vec[i] = u_D[i]
    for i in free_nodes:
        for j in boundary_nodes:
            b_vec[i] -= A_global[i, j] * u_D[j]
            A_global[i, j] = 0
    
    # Solve the linear system.
    u = np.linalg.solve(A_global, b_vec)
    return u, K_global, M_global


def PlotSolutionHelmoltz(u, N, T):
    """
    Plot the FE solution over the triangulation.
    """
    import matplotlib.tri as mtri
    triangles = T[:, :3]
    triang = mtri.Triangulation(N[:, 0], N[:, 1], triangles)
    plt.figure()
    plt.tripcolor(triang, np.real(u), shading='gouraud')
    plt.colorbar()
    plt.title('FE Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()

def g(x, y):
    # Use a tolerance when comparing x to 0.5.
    if abs(x - 0.5) < 1e-3 and (0.1 <= y <= 0.2):
        return 1.0
    else:
        return 0.0

def compare_exact():
    u_exact = N[:, 0] + N[:, 1]
    error = np.linalg.norm(u - u_exact, ord=float('inf'))
    print("Max error compared to exact solution:", error)

# -----------------------
# Example usage:
if __name__ == "__main__":
    # Solve the Laplace (Helmholtz with ω=0) problem on the unit square (G=0)
    # with Dirichlet condition g(x,y) = x+y.
    # For ω=0 the PDE reduces to -Δu = 0; the exact solution is u(x,y)=x+y.
    w = 2 * np.pi * 2.45e9  # 2.45 GHz in rad/s
    #G = 0 # choose unit square
    #N, T, P = mygrid(G, w, points=50)
    #g = lambda x, y: x + y
    N,T,P = chicken_mesh(w)
    # Plot the generated mesh.
    plotmygrid(N, T, P)
    
    u, K, M = FEHelmholtz2D(g, N, T, w, P)

    PlotSolutionHelmoltz(u, N, T)
    
    #compare_exact()