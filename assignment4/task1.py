import numpy as np
import matplotlib.pyplot as plt

def fem_conv_diff(D, m):
    """
    Solve the 1D convection-diffusion problem:
        -D u''(x) + beta u'(x) = 1,  for x in (0, pi),
    with Dirichlet boundary conditions u(0)=0, u(pi)=0,
    using a uniform mesh of m+1 elements and CG(1) finite elements.
    
    Parameters
    ----------
    D : float
        Diffusion coefficient (positive).
    beta : float
        Convection coefficient in front of u'(x).
    m : int
        Number of subintervals minus 1. The mesh has (m+1) subintervals.
    
    Returns
    -------
    x : numpy.ndarray
        Array of mesh node coordinates (size m+2).
    U : numpy.ndarray
        The finite element solution at each mesh node (size m+2).
    """
    # Domain
    a = 0.0
    b = np.pi
    
    # Mesh 
    h = (b - a) / (m + 1)
    
    # Node coordinates
    x = np.linspace(a, b, m+2)
    
    # Matrices
    K_diff = (D/h) * np.array([[ 1, -1],
                               [-1,  1]])
    
    K_conv = (1/4)*np.array([[-1,  1],
                       [-1,  1]])
    
    
    K_mat = K_diff + K_conv
    
    
    f_mat = np.array([h/2, h/2])
    
    # Initialize whole matrix and RHS
    A = np.zeros((m+2, m+2))
    F = np.zeros(m+2)
    
    # Assembly loop over each element e
    for e in range(m+1):
        i = e
        j = e + 1
        
        # Add local contributions to global matrix
        A[i, i] += K_mat[0, 0]
        A[i, j] += K_mat[0, 1]
        A[j, i] += K_mat[1, 0]
        A[j, j] += K_mat[1, 1]
        
        # Add local contributions to global RHS
        F[i] += f_mat[0]
        F[j] += f_mat[1]
    
    # Impose Dirichlet boundary conditions: u(0)=0, u(pi)=0
    # Overwrite row 0
    A[0, :] = 0
    A[0, 0] = 1
    F[0] = 0
    
    # Overwrite row m+1
    A[m+1, :] = 0
    A[m+1, m+1] = 1
    F[m+1] = 0
    
    # Solve linear system
    U = np.linalg.solve(A, F)
    
    return x, U

def exact_solution(D, x):
    return (2*np.pi / (np.exp(np.pi/(2*D)) - 1)) * (1 - np.exp(x/(2*D))) + 2*x



def compute_error(D, M):
    """
    Solve the FEM approximation for given D, M, and compare to exact solution.
    
    Parameters
    ----------
    D : float
        Diffusion coefficient.
    M : int
        Number of elements in [0, pi].
    norm_type : str
        'max' for max-norm error, or 'L2' for a discrete L2 error.
    
    Returns
    -------
    error : float
        The computed error between the FEM and exact solution.
    """
    x, u_fem = fem_conv_diff(D, M)

    u_ex = exact_solution(D, x)
    
    # Quick discrete approximation we just do the sum*(h).
    h = np.pi / M
    return np.sqrt( np.sum((u_fem - u_ex)**2) * h )

def run_experiments():
    """
    Compare FEM approximation and exact solution for two regimes:
    1) Pe ~ 1  => D ~1
    2) Pe >> 1 => D << 1
    """

    Ds = [10,1.0, 0.01]
    Ms = [10, 20, 40, 80, 160, 320] 

    results = {}
    for D in Ds:
        errs = []
        for M in Ms:
            err = compute_error(D, M)
            errs.append(err)
        results[D] = (Ms, errs)

   
    for D in Ds:
        Ms, errs = results[D]
        print(f"\nResults for D = {D} (Pe ~ {np.pi/(2*D):.2f}):")
        for (m, e) in zip(Ms, errs):
            print(f"  M={m:4d}, h={np.pi/m:.5f}, max error={e:.4e}")

    # Plot error vs. 1/M for each diffusion term
    plt.figure(figsize=(7,5))
    for D in Ds:
        Ms, errs = results[D]
        plt.plot(Ms, errs, 'o--', label=f"D={D} (Pe~{np.pi/(2*D):.1f})")
    plt.xlabel("Number of elements (M)")
    plt.ylabel("Max-norm error")
    plt.title("Error vs. mesh refinement for different D")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    D = 0.01
    M = 4
    x, U = fem_conv_diff(D, M)

    
    x_fine = np.linspace(0, np.pi, 3000)
    u_exact = exact_solution(D, x_fine)
    plt.figure()
    plt.plot(x_fine, u_exact, 'k-', label='Exact solution')
    plt.plot(x, U, 'ro-', label='FEM solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()
    run_experiments()
