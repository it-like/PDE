import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def mygrid(G, w, num_points=21):
    """
    Generate a triangulation of the computational domain.
    G: integer (0,1,2,3) selects the geometry:
       0: unit square [0,1]x[0,1]
       1: a single triangle (vertices (0,0), (1,0), (0,1))
       2: rectangle [0,2]x[0,1]
       3: "chicken in a microwave" on unit square,
          with a chicken region (an ellipse) having different material constant.
    w: frequency (not used in grid generation, but required by the signature)
    num_points: number of grid points in each direction for structured grids.
    Returns:
       N: (n_nodes,2) array of node coordinates.
       T: (n_tri,6) integer array; first 3 columns: node indices (0-indexed);
          next 3 columns: boundary flags (1 if the corresponding edge is on the domain boundary, 0 otherwise).
       P: (n_tri,) array with the material constant (mu*epsilon) for each triangle.
    """
    tol = 1e-6
    if G == 0:
        # Unit square [0,1]x[0,1]
        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        X, Y = np.meshgrid(x, y)
        points = np.vstack((X.flatten(), Y.flatten())).T
        tri = Delaunay(points)
        T_list = []
        for tri_indices in tri.simplices:
            flags = []
            # For each edge, mark as boundary if both endpoints lie on one of the four sides.
            for i in range(3):
                j = (i+1) % 3
                p1 = points[tri_indices[i]]
                p2 = points[tri_indices[j]]
                flag = 0
                if (abs(p1[0]) < tol and abs(p2[0]) < tol) or \
                   (abs(p1[0]-1) < tol and abs(p2[0]-1) < tol) or \
                   (abs(p1[1]) < tol and abs(p2[1]) < tol) or \
                   (abs(p1[1]-1) < tol and abs(p2[1]-1) < tol):
                    flag = 1
                flags.append(flag)
            T_list.append(np.concatenate([tri_indices, np.array(flags)]))
        T = np.array(T_list, dtype=int)
        # Air: use material constants μₗ and εₗ.
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        P = np.full(len(T), mu_l * eps_l)
        return points, T, P

    elif G == 1:
        # A single triangle with vertices (0,0), (1,0), (0,1)
        points = np.array([[0, 0], [1, 0], [0, 1]])
        # All three edges are on the boundary.
        T = np.array([[0, 1, 2, 1, 1, 1]])
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        P = np.array([mu_l * eps_l])
        return points, T, P

    elif G == 2:
        # Rectangle: [0,2]x[0,1]
        x = np.linspace(0, 2, num_points)
        y = np.linspace(0, 1, num_points)
        X, Y = np.meshgrid(x, y)
        points = np.vstack((X.flatten(), Y.flatten())).T
        tri = Delaunay(points)
        T_list = []
        for tri_indices in tri.simplices:
            flags = []
            for i in range(3):
                j = (i+1) % 3
                p1 = points[tri_indices[i]]
                p2 = points[tri_indices[j]]
                flag = 0
                if (abs(p1[0]) < tol and abs(p2[0]) < tol) or \
                   (abs(p1[0]-2) < tol and abs(p2[0]-2) < tol) or \
                   (abs(p1[1]) < tol and abs(p2[1]) < tol) or \
                   (abs(p1[1]-1) < tol and abs(p2[1]-1) < tol):
                    flag = 1
                flags.append(flag)
            T_list.append(np.concatenate([tri_indices, np.array(flags)]))
        T = np.array(T_list, dtype=int)
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        P = np.full(len(T), mu_l * eps_l)
        return points, T, P

    elif G == 3:
        # "Chicken in a microwave": Domain is unit square [0,1]x[0,1].
        # Generate a structured grid.
        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        X, Y = np.meshgrid(x, y)
        points = np.vstack((X.flatten(), Y.flatten())).T
        tri = Delaunay(points)
        T_list = []
        for tri_indices in tri.simplices:
            flags = []
            for i in range(3):
                j = (i+1) % 3
                p1 = points[tri_indices[i]]
                p2 = points[tri_indices[j]]
                flag = 0
                if (abs(p1[0]) < tol and abs(p2[0]) < tol) or \
                   (abs(p1[0]-1) < tol and abs(p2[0]-1) < tol) or \
                   (abs(p1[1]) < tol and abs(p2[1]) < tol) or \
                   (abs(p1[1]-1) < tol and abs(p2[1]-1) < tol):
                    flag = 1
                flags.append(flag)
            T_list.append(np.concatenate([tri_indices, np.array(flags)]))
        T = np.array(T_list, dtype=int)
        # Define material constants:
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        mu_p = 4 * np.pi * 1e-7
        eps_p = 4.43e-11
        # For each triangle, use its centroid to decide whether it is in the "chicken" (inside an ellipse) or in air.
        P = []
        for tri_indices in T:
            verts = points[tri_indices[:3]]
            centroid = np.mean(verts, axis=0)
            # Define chicken region as an ellipse with center (0.5,0.5) and radii 0.2 and 0.3.
            if ((centroid[0]-0.5)/0.2)**2 + ((centroid[1]-0.5)/0.3)**2 <= 1:
                P.append(mu_p * eps_p)
            else:
                P.append(mu_l * eps_l)
        P = np.array(P)
        return points, T, P

def plotmygrid(N, T, P):
    """
    Plot the triangulation.
    The triangles are drawn in light gray and the boundary edges (those with flag==1) are drawn in bold.
    If the number of nodes is less than 20, node numbers are also plotted.
    """
    plt.figure()
    # Plot each triangle (with its three vertices)
    for tri in T:
        verts = N[tri[:3]]
        poly = plt.Polygon(verts, edgecolor='gray', facecolor='none')
        plt.gca().add_patch(poly)
        # Draw boundary edges in black and thicker.
        for i in range(3):
            if tri[3+i] == 1:
                j = (i+1) % 3
                edge = N[tri[[i, j]]]
                plt.plot(edge[:, 0], edge[:, 1], 'k-', linewidth=2)
    # Optionally also draw a light triplot.
    plt.triplot(N[:, 1], N[:, 0], T[:, :3], color='lightgray', linestyle='--')
    plt.scatter(N[:, 0], N[:, 1], color='red', s=10)
    if N.shape[0] < 20:
        for i, (x, y) in enumerate(N):
            plt.text(x, y, str(i+1), color='blue', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mesh')
    plt.show()

def mygridrefinement(N, T, P):
    """
    Refine the grid by splitting each triangle into 4 subtriangles.
    New nodes are added at the midpoint of each edge.
    Returns refined nodes, triangles and material property array.
    """
    new_nodes = N.tolist()
    node_dict = {}  # to store midpoints: key = (min_index, max_index), value = new node index
    new_T = []
    new_P = []

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

    # Loop over each triangle and subdivide.
    for k, tri in enumerate(T):
        indices = tri[:3]
        i1, i2, i3 = indices
        m12 = get_midpoint(i1, i2)
        m23 = get_midpoint(i2, i3)
        m31 = get_midpoint(i3, i1)

        # Define the 4 new triangles.
        def compute_flags(tri_indices):
            pts = np.array([new_nodes[i] for i in tri_indices])
            flags = []
            for i in range(3):
                j = (i+1) % 3
                p1 = pts[i]
                p2 = pts[j]
                flag = 0
                if (abs(p1[0] - np.min(N[:,0])) < tol and abs(p2[0] - np.min(N[:,0])) < tol) or \
                   (abs(p1[0] - np.max(N[:,0])) < tol and abs(p2[0] - np.max(N[:,0])) < tol) or \
                   (abs(p1[1] - np.min(N[:,1])) < tol and abs(p2[1] - np.min(N[:,1])) < tol) or \
                   (abs(p1[1] - np.max(N[:,1])) < tol and abs(p2[1] - np.max(N[:,1])) < tol):
                    flag = 1
                flags.append(flag)
            return flags

        tol = 1e-6
        tri1 = np.array([i1, m12, m31])
        tri2 = np.array([m12, i2, m23])
        tri3 = np.array([m31, m23, i3])
        tri4 = np.array([m12, m23, m31])
        new_T.append(np.concatenate([tri1, compute_flags(tri1)]))
        new_T.append(np.concatenate([tri2, compute_flags(tri2)]))
        new_T.append(np.concatenate([tri3, compute_flags(tri3)]))
        new_T.append(np.concatenate([tri4, compute_flags(tri4)]))
        new_P += [P[k]] * 4

    new_N = np.array(new_nodes)
    new_T = np.array(new_T, dtype=int)
    new_P = np.array(new_P)
    return new_N, new_T, new_P

def elementstiffmatrix(t):
    """
    Compute the element stiffness matrix for a triangle.
    t: (3,2) array with the coordinates of the triangle vertices.
    Returns:
       Se: 3x3 element stiffness matrix.
       A: the area of the triangle.
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
    Compute the element mass matrix for a triangle given its area A.
    For linear (P1) elements, the mass matrix is (A/12)*[[2,1,1],[1,2,1],[1,1,2]].
    """
    Me = (A / 12.0) * np.array([[2, 1, 1],
                                [1, 2, 1],
                                [1, 1, 2]])
    return Me

def FEHelmholtz2D(g, N, T, w, P):
    """
    Solve the Helmholtz equation in 2D: Δu + ω²*(μ*ε)*u = 0 in the domain,
    with Dirichlet boundary condition u = g on the boundary.
    The weak form leads to: K*u - ω²*M*u = 0.
    Here the matrices are assembled elementwise.
    g: function g(x,y) prescribing the Dirichlet condition.
    N: nodes array.
    T: triangle connectivity (first 3 columns are node indices).
    w: frequency ω.
    P: array of material constants for each triangle.
    Returns:
       u: the FEM solution at the nodes.
       K_global, M_global: the assembled stiffness and (weighted) mass matrices.
    """
    n_nodes = N.shape[0]
    K_global = np.zeros((n_nodes, n_nodes))
    M_global = np.zeros((n_nodes, n_nodes))
    # Assemble element matrices.
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
    # Identify boundary nodes using the boundary flags stored in T.
    boundary_nodes = set()
    for tri in T:
        for i in range(3):
            if tri[3+i] == 1:
                boundary_nodes.add(tri[i])
    boundary_nodes = np.array(list(boundary_nodes))
    # Prescribe Dirichlet data.
    u_D = np.zeros(n_nodes)
    for i in boundary_nodes:
        u_D[i] = g(N[i, 0], N[i, 1])
    # Build right-hand side vector b (initially zero).
    b_vec = np.zeros(n_nodes)
    # Impose Dirichlet BCs: for each boundary node, set the corresponding row to identity.
    free_nodes = np.setdiff1d(np.arange(n_nodes), boundary_nodes)
    for i in boundary_nodes:
        A_global[i, :] = 0
        A_global[i, i] = 1
        b_vec[i] = u_D[i]
    # For free nodes, subtract the contribution from Dirichlet nodes.
    for i in free_nodes:
        for j in boundary_nodes:
            b_vec[i] -= A_global[i, j] * u_D[j]
            A_global[i, j] = 0
    # Solve the linear system.
    u = np.linalg.solve(A_global, b_vec)
    return u, K_global, M_global

def PlotSolutionHelmoltz(u, N, T):
    """
    Plot the numerical solution u over the triangulation.
    """
    import matplotlib.tri as mtri
    triangles = T[:, :3]
    triang = mtri.Triangulation(N[:, 0], N[:, 1], triangles)
    plt.figure()
    plt.tripcolor(triang, u, shading='gouraud')
    plt.colorbar()
    plt.title('FE Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()

# ---------------------------------------------------
# Example of use:
if __name__ == "__main__":
    # Test case: Solve the Laplace (Helmholtz with ω=0) problem on the unit square (G=0)
    # with Dirichlet condition g(x,y) = x+y.
    # (For ω = 0 the PDE reduces to -Δu = 0; the exact solution is u(x,y)=x+y.)
    g = lambda x, y: x + y
    w = 2*np.pi*2.45e9  # frequency zero => Laplace problem
    G = 0    # choose unit square
    N, T, P = mygrid(G, w, num_points=2)
    
    # Plot the generated mesh.
    plotmygrid(N, T, P)
    exit()
    # Solve the FE problem.
    u, K, M = FEHelmholtz2D(g, N, T, w, P)
    
    # Plot the FE solution.
    PlotSolutionHelmoltz(u, N, T)
    
    # Compare with the exact solution u_exact = x+y.
    u_exact = N[:, 0] + N[:, 1]
    error = np.linalg.norm(u - u_exact, ord=np.inf)
    print("Max error compared to exact solution:", error)
