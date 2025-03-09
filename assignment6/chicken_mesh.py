import numpy as np

def chicken_mesh(w):
    """
    Create the mesh and material properties for the "chicken in a microwave" problem.
    
    Parameters:
      w : float
          The frequency (in rad/s). Must be nonzero.
    
    Returns:
      N : (40,2) numpy array
          Array of node coordinates.
      T : (n_tri,6) numpy array (int)
          Triangle connectivity. The first three entries in each row are the node indices (0-indexed),
          and the next three are boundary flags (0 or 1).
      P : (n_tri,) numpy array (complex)
          Material property for each triangle. For the first 29 triangles (the chicken region) this is
          muChicken*(epsChicken + sigmaChicken/(1j*w)) and for the remaining 34 triangles (air) it is
          muAir*epsAir.
    """
    # Physical constants
    eps_air = 8.85e-12
    mu_air = 4 * np.pi * 1e-7
    sigma_air = 0.0

    eps_chicken = 4.43e-11
    mu_chicken = 4 * np.pi * 1e-7
    sigma_chicken = 3e-11

    # Define the nodes (40×2)
    N = np.array([
        [0.08, 0.09],
        [0.14, 0.04],
        [0.2,  0.03],
        [0.3,  0.03],
        [0.35, 0.04],
        [0.4,  0.1],
        [0.35, 0.16],
        [0.3,  0.2],
        [0.25, 0.22],
        [0.2,  0.24],
        [0.14, 0.24],
        [0.1,  0.2],
        [0.05, 0.2],
        [0.03, 0.22],
        [0.02, 0.19],
        [0.03, 0.16],
        [0.05, 0.18],
        [0.1,  0.18],
        [0.13, 0.14],
        [0.15, 0.18],
        [0.2,  0.2],
        [0.15, 0.1],
        [0.2,  0.1],
        [0.3,  0.1],
        [0.35, 0.1],
        [0.0,  0.0],
        [0.1,  0.0],
        [0.2,  0.0],
        [0.3,  0.0],
        [0.4,  0.0],
        [0.5,  0.0],
        [0.5,  0.3],
        [0.4,  0.3],
        [0.25, 0.3],
        [0.15, 0.3],
        [0.08, 0.3],
        [0.0,  0.3],
        [0.0,  0.19],
        [0.0,  0.1],
        [0.5,  0.15]
    ])
    

    T_raw = np.array([
        [13, 14, 15, 0, 0, 0],
        [15, 16, 17, 0, 0, 0],
        [15, 17, 13, 0, 0, 0],
        [17, 12, 13, 0, 0, 0],
        [17, 18, 12, 0, 0, 0],
        [18, 20, 12, 0, 0, 0],
        [18, 19, 20, 0, 0, 0],
        [12, 20, 11, 0, 0, 0],
        [20, 21, 11, 0, 0, 0],
        [21, 10, 11, 0, 0, 0],
        [21, 9, 10, 0, 0, 0],
        [1, 22, 19, 0, 0, 0],
        [22, 20, 19, 0, 0, 0],
        [22, 23, 20, 0, 0, 0],
        [2, 22, 1, 0, 0, 0],
        [2, 23, 22, 0, 0, 0],
        [20, 23, 21, 0, 0, 0],
        [23, 9, 21, 0, 0, 0],
        [2, 3, 23, 0, 0, 0],
        [3, 24, 23, 0, 0, 0],
        [23, 24, 9, 0, 0, 0],
        [3, 4, 24, 0, 0, 0],
        [24, 8, 9, 0, 0, 0],
        [24, 7, 8, 0, 0, 0],
        [24, 25, 7, 0, 0, 0],
        [24, 5, 25, 0, 0, 0],
        [4, 5, 24, 0, 0, 0],
        [5, 6, 25, 0, 0, 0],
        [6, 7, 25, 0, 0, 0],
        [26, 27, 1, 1, 0, 0],
        [27, 2, 1, 0, 0, 0],
        [27, 28, 2, 1, 0, 0],
        [28, 3, 2, 0, 0, 0],
        [28, 29, 3, 1, 0, 0],
        [29, 4, 3, 0, 0, 0],
        [29, 30, 4, 1, 0, 0],
        [30, 5, 4, 0, 0, 0],
        [30, 6, 5, 0, 0, 0],
        [30, 31, 6, 1, 0, 0],
        [31, 40, 6, 1, 0, 0],
        [32, 33, 40, 1, 0, 1],
        [33, 7, 6, 0, 0, 0],
        [33, 8, 7, 0, 0, 0],
        [33, 34, 8, 1, 0, 0],
        [34, 9, 8, 0, 0, 0],
        [34, 10, 9, 0, 0, 0],
        [34, 35, 10, 1, 0, 0],
        [35, 11, 10, 0, 0, 0],
        [35, 36, 11, 1, 0, 0],
        [36, 12, 11, 0, 0, 0],
        [36, 13, 12, 0, 0, 0],
        [36, 14, 13, 0, 0, 0],
        [36, 37, 14, 1, 0, 0],
        [37, 38, 14, 1, 0, 0],
        [38, 15, 14, 0, 0, 0],
        [38, 16, 15, 0, 0, 0],
        [38, 39, 16, 1, 0, 0],
        [39, 1, 16, 0, 0, 0],
        [1, 17, 16, 0, 0, 0],
        [1, 18, 17, 0, 0, 0],
        [1, 19, 18, 0, 0, 0],
        [39, 26, 1, 1, 0, 0],
        [40, 33, 6, 0, 0, 0]
    ], dtype=int)
    # Adjust the first three columns to 0-indexing:
    T = T_raw.copy()
    T[:, :3] = T_raw[:, :3] - 1

   
    P_chicken = mu_chicken * (eps_chicken + sigma_chicken/(1j * w))
    P_air = mu_air * eps_air
    # Concatenate into a single vector (first 29 triangles then 34 triangles)
    P = np.hstack((P_chicken * np.ones(29), P_air * np.ones(34)))
    return N, T, P


