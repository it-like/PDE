import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from chicken_mesh import chicken_mesh  # Provided module for the chicken geometry

def myGrid(G: int, ω: float):

    """
    Generate triangulation (N,T,P) where G dictates what type:
    G == 0: Unit square
    G == 1: Triangle
    G == 2: Rectangle (empty microwave)
    G == 3: Chicken in microwave
    Args:
        G (int): flag for plotting
        ω (float): frequency of Helmholtz equation
    Returns:
        N: numpy array of nodes (each row is [x,y])
        T: numpy array of triangles. For each triangle, first 3 entries are node indices and the last 3 are boundary flags.
        P: Material constants (μ·ε) for each triangle
    """
    if G not in [0, 1, 2, 3]:
        raise Exception(f"G has to be integer between and including 0 and 3, was {G}")
    
    # Unit square case
    if G == 0:
        N = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Nodes
        T = np.array([
            [0, 1, 2, 1, 0, 1],  # triangle #1: edges (0-1,1-2,2-0)
            [1, 3, 2, 1, 1, 0]   # triangle #2: edges (1-3,3-2,2-1)
        ])
        P = [1, 1]
    
    # Single triangle in unit square case
    if G == 1 or G == 2:
        N = np.array([[0, 0], [1, 0], [0, 1]])
        T = np.array([[0, 1, 2, 1, 1, 0]])  # all three edges on boundary
        mu_l = 4 * np.pi * 1e-7
        eps_l = 8.85e-12
        P = np.array([mu_l * eps_l])

    if G == 3:
        return chicken_mesh(ω)

    return N, T, P

def myGridRefinement(N, T, P):
    """
    Refine the grid by splitting each triangle into 4 subtriangles.
    New nodes are added at the midpoints of each edge.
    Args:
        N: numpy array of nodes (x,y)
        T: numpy array of triangles (first 3 columns: node indices, last 3: boundary flags)
        P: Material constants for each triangle
    Returns:
        Nr: refined nodes
        Tr: refined triangle connectivity (with inherited boundary flags)
        Pr: refined material property array
    """
    Nr = N.copy().tolist()
    node_to_index = {tuple(pt): idx for idx, pt in enumerate(Nr)}
    nn = len(Nr)
    Tr = []
    Pr = []

    # Use this to see if the node exists already
    def find_or_add_node(coords):
        nonlocal nn
        key = tuple(coords)
        if key in node_to_index:
            return node_to_index[key]
        else:
            node_to_index[key] = nn
            Nr.append(list(coords))
            nn += 1
            return node_to_index[key]

    for j in range(len(T)):
        i1, i2, i3 = T[j, :3].astype(int)
        b1, b2, b3 = T[j, 3:]
        p1 = N[i1]
        p2 = N[i2]
        p3 = N[i3]
        n4 = (p1 + p2) / 2  
        n5 = (p1 + p3) / 2  
        n6 = (p2 + p3) / 2  
        i4 = find_or_add_node(n4) # Check if node exists 
        i5 = find_or_add_node(n5) # Check if node exists
        i6 = find_or_add_node(n6) # Check if node exists
        tri1 = [i1, i4, i5, b1, 0, b3]
        tri2 = [i4, i6, i5, 0, 0, 0]
        tri3 = [i5, i6, i3, 0, b2, b3]
        tri4 = [i4, i2, i6, b1, b2, 0]
        
        Tr.append(tri1)
        Tr.append(tri2)
        Tr.append(tri3)
        Tr.append(tri4)
        Pr.extend([P[j]] * 4)
    Nr = np.array(Nr)
    Tr = np.array(Tr)
    Pr = np.array(Pr)
    return Nr, Tr, Pr


def element_stiffness_matrix(t):
    """
    Computes the element stiffness matrix for a linear triangle.
    t: 3x2 array with node coordinates [[x1,y1], [x2,y2], [x3,y3]]
    Returns:
        Se: 3x3 stiffness matrix
    """
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    # Compute triangle area using the determinant formula
    A = 0.5 * abs(np.linalg.det(np.array([[1, x1, y1],
                                           [1, x2, y2],
                                           [1, x3, y3]])))
    # Compute coefficients for the gradients of the shape functions:
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    Se = np.zeros((3, 3)) # Stiffness element
    for i in range(3):
        for j in range(3):
            Se[i, j] = (b[i]*b[j] + c[i]*c[j]) / (4 * A) # Change of variable formula
    return Se

def element_mass_matrix(t):
    """
    Computes the element mass matrix for a linear triangle.
    t: 3x2 array with node coordinates
    Returns:
        Me: 3x3 mass matrix computed as (A/12)*[[2,1,1],[1,2,1],[1,1,2]]
    """
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    A = 0.5 * abs(np.linalg.det(np.array([[1, x1, y1],
                                           [1, x2, y2],
                                           [1, x3, y3]])))
    Me = (A / 12.0) * np.array([[2, 1, 1],
                                [1, 2, 1],
                                [1, 1, 2]])
    return Me

def FEHelmholtz2D(g, N, T, w, P):
    """
    Finite element solver for the Helmholtz equation:
      Δu + w² * P * u = 0  in the domain,
      u = g on the boundary of the domain
    
    Inputs:
      g: Dirichlet boundary function g(x,y)
      N: array of nodes (n_nodes x 2)
      T: array of triangles (n_triangles x 6) where the first 3 columns are node indices,
         and the last 3 indicate if the corresponding edge is on the boundary (1) or not (0).
      w: frequency (float)
      P: material constant for each triangle (array of length n_triangles)
    
    Returns:
      u: solution vector (nodal values)
      A_global: assembled global matrix
      b_global: global load vector (nonzero only for Dirichlet nodes)
    """
    n_nodes = N.shape[0]
    A_global = np.zeros((n_nodes, n_nodes))
    b_global = np.zeros(n_nodes)
    
    # Assemble element contributions into global matrix
    for e in range(T.shape[0]):
        nodes = T[e, :3].astype(int)
        coords = N[nodes, :]  # 3x2 array of coordinates
        Se = element_stiffness_matrix(coords)
        Me = element_mass_matrix(coords)
        Ae = Se - (w**2) * P[e] * Me
        for i_local, i_global in enumerate(nodes): # Add together all element into global
            for j_local, j_global in enumerate(nodes):
                A_global[i_global, j_global] += Ae[i_local, j_local]
                
    boundary_nodes = set()
    for tri in T:
        nodes = tri[:3].astype(int)
        edges = [(nodes[0], nodes[1], tri[3]),
                 (nodes[1], nodes[2], tri[4]),
                 (nodes[2], nodes[0], tri[5])]
        for (i, j, flag) in edges:
            if flag == 1:
                boundary_nodes.add(i)
                boundary_nodes.add(j)
    boundary_nodes = list(boundary_nodes)
    
    for i in boundary_nodes:
        A_global[i, :] = 0
        A_global[i, i] = 1
        b_global[i] = g(N[i, 0], N[i, 1])
    
    u = np.linalg.solve(A_global, b_global)
    return u, A_global, b_global


def plotFEM(N, T, u=None, P=None,
            show_solution=False,
            show_grid=False,
            rectangular=False,
            node_labels=True,
            title=None,
            cmap='viridis'):
    """
    Unified plotting function for FEM grids and solutions.

    Parameters
    ----------
    N : ndarray
        Node coordinates of shape (n_nodes, 2).
    T : ndarray
        Triangles of shape (n_triangles, >=3).
        The first 3 columns are node indices.
        If show_grid=True, the last 3 columns are boundary flags (0 or 1).
    u : ndarray or None
        Nodal solution vector (length = number of nodes).
        If None, no solution is plotted.
    P : ndarray or None
        Material constants (unused here, but left for completeness).
    show_solution : bool
        If True, display tricontourf of the solution `u`.
    show_grid : bool
        If True, overlay the grid edges, with bold edges for boundary flags=1.
    rectangular : bool
        If True, use an 'auto' aspect ratio (rectangular).
        If False, use an 'equal' aspect ratio (square).
    node_labels : bool
        If True, label the nodes with their index. (Or only if few nodes.)
    title : str or None
        Title for the figure.
    cmap : str
        Colormap name for tricontourf (default 'viridis').
    """
    plt.figure(figsize=(8, 4))

    if show_solution and (u is not None):
        triangles = T[:, :3].astype(int)
        triang = mtri.Triangulation(N[:, 0], N[:, 1], triangles)
        plt.tricontourf(triang, u, cmap=cmap)
        plt.colorbar(label="Solution Value")


    if show_grid:
        visualized_edges = set()
        do_labels = node_labels and (len(N) <= 25)

        for tri in T:
            nodes = tri[:3].astype(int)
            b_flags = tri[3:] 

            for i in range(3):
                edge = tuple(sorted((nodes[i], nodes[(i+1) % 3])))
                if edge not in visualized_edges:
                    # thick line if boundary=1, thin otherwise
                    linewidth = 3.0 if (b_flags[i] == 1) else 0.3
                    plt.plot([N[edge[0], 0], N[edge[1], 0]],
                             [N[edge[0], 1], N[edge[1], 1]],
                             linewidth=linewidth, color='b')
                    visualized_edges.add(edge)

        # label nodes, max to 25 nodes
        if do_labels:
            for i, (x, y) in enumerate(N):
                plt.text(x, y, f'{i+1}', color='red', ha='center', va='center')

    if rectangular:
        plt.gca().set_aspect('auto', adjustable='box')
    else:
        plt.axis('equal')

    if title:
        plt.title(title)
        svg_filename = "assignment6/images/" + title.replace(" ", "_") + ".svg"
    else:
        svg_filename = "plotFEM_output.svg"

    plt.savefig(svg_filename, format="svg")
    plt.show()
    print(f"Figure saved to '{svg_filename}'")

if __name__ == "__main__":
    import numpy as np

    # Example I: Unit Square Test 
    ω_0 = 0.0
    N, T, P = myGrid(0, ω_0)  # G=0 -> unit square
    refine = 6
    #for _ in range(refine): # Refine grid
    #    N, T, P = myGridRefinement(N, T, P)

    # Dirichlet BC for the square: g = x + y
    g_square = lambda x, y: x + y
    u_square, A_sq, b_sq = FEHelmholtz2D(g_square, N, T, ω_0, P)

    # Plot the solution on the square 
    #plotFEM(N, T, u=u_square, P=P, show_solution=True, show_grid=True,
    #        rectangular=False, title=f"Solution Refined {refine} times on Unit Square")

    # Compare numerical solution with the exact solution
    u_exact = N[:, 0] + N[:, 1]
    error = np.linalg.norm(u_square - u_exact, ord=np.inf)
    print("Max error for unit square test (should be near 0):", error)



    # Example II: Chicken in Microwave
    ω_chicken = 2 * np.pi * 2.45e9
    N_chicken, T_chicken, P_chicken = myGrid(3, ω_chicken)  # G=3 -> chicken
    refine = 5
    for _ in range(refine): # Refine mesh
        N_chicken, T_chicken, P_chicken = myGridRefinement(N_chicken, T_chicken, P_chicken)


    #plotFEM(N_chicken, T_chicken, P=P_chicken, show_grid=True,
    #        rectangular=True, title=f"{refine} times Refined Chicken Mesh")

    # Dirichlet BC for chicken
    def g_chicken(x, y):
        if np.isclose(x, 0.5, atol=1e-2) and (0.1 <= y <= 0.2):
            return 100.0
        return 0.0

    u_chicken, A_ch, b_ch = FEHelmholtz2D(g_chicken, N_chicken, T_chicken,
                                          ω_chicken, P_chicken)
    # Plot chicken FEM
    plotFEM(N_chicken, T_chicken, u=u_chicken, P=P_chicken,
            show_solution=True, show_grid=False,
            rectangular=True, title=f"{refine} times Refined Chicken in Microwave Without Mesh")

