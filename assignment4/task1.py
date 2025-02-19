import numpy as np
import matplotlib.pyplot as plt

def fem_conv_diff(D, m):
    
    a = 0.0
    b = np.pi
    
    # Mesh 
    h = (b - a) / (m + 1)
    
    # hat_x_val
    x = np.linspace(a, b, m+2) # m points in mesh, two on boundary
    
    # See math before
    K_diff = (D/h) * np.array([[ 1, -1],
                               [-1,  1]])
    
    K_conv = (1/4)*np.array([[-1,  1],
                       [-1,  1]])
    
    
    K_mat = K_diff + K_conv
    
    # RHS
    f_mat = (h/2)*np.ones(2)
    
    # Initialize RHS and RHS, +2 for the edges
    A = np.zeros((m+2, m+2))
    F = np.zeros(m+2)
    

    for e in range(m+1):
        i = e
        j = e + 1

        A[i, i] += K_mat[0, 0]
        A[i, j] += K_mat[0, 1]
        A[j, i] += K_mat[1, 0]
        A[j, j] += K_mat[1, 1]
        
        F[i] += f_mat[0]
        F[j] += f_mat[1]

    
    # Impose Dirichlet boundary conditions: u(0)=0, u(pi)=0:

    # Overwrite row 0
    
    A[0, :] = 0
    A[:, 0] = 0
    A[0, 0] = 1 # Force U(0) = 0 
    F[0] = 0    # By AU_0 = F_0 => U_0 = 0
    
    # Overwrite row m+1
    A[m+1, :] = 0
    A[:, m+1] = 0
    A[m+1, m+1] = 1 # Force U(1) = 0 
    F[m+1] = 0      # By AU_{m+1} = F_{m+1} => U# By AU = F => U = 0 
    # Solve linear system
    U = np.linalg.solve(A, F)
    
    return x, U

def exact_solution(D, x):
    return (2*np.pi / (np.exp(np.pi/(2*D)) - 1)) * (1 - np.exp(x/(2*D))) + 2*x  


def compute_error(D, M):
    x_coarse, u_fem = fem_conv_diff(D, M)
    
    N_fine = 10000
    x_fine = np.linspace(0, np.pi, N_fine)
    
    
    u_ex_fine = exact_solution(D, x_fine)
    
    u_fem_interp = np.interp(x_fine, x_coarse, u_fem)
    h_fine = (np.pi - 0) / (N_fine - 1)
    error = np.sqrt(np.sum((u_fem_interp - u_ex_fine)**2) * h_fine)
    return error

def run_experiments():
    Ds = [np.round(np.pi/2,4), 0.01]
    Ms = np.arange(100)

    results = {}
    for D in Ds:
        errs = []
        for M in Ms:
            err = compute_error(D, M)
            errs.append(err)
        results[D] = (Ms, errs)

   
    plt.figure(figsize=(7,5))
    for D in Ds:
        Ms, errs = results[D]
        plt.plot(Ms, errs, 'o--', label=f"D={D} (Pe~{np.pi/(2*D):.1f})")
    plt.xlabel("Number of elements (M)")
    plt.ylabel("Error")
    plt.title("Error vs. mesh refinement for different D")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    #plt.savefig("assignment4/images/Pe_approx.svg", format="svg")
    plt.show()
    


if __name__ == "__main__":
    D = 1
    M = 12
    x, U= fem_conv_diff(D, M)
    print(np.round(U,3))

    
    x_fine = np.linspace(0, np.pi, 3000)
    u_exact = exact_solution(D, x_fine)
    plt.figure()
    plt.grid(True)
    plt.plot(x_fine, u_exact, 'k-', label='Exact solution')
    plt.plot(x, U, 'ro-', label='FEM solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.savefig("assignment4/images/mesh_12_D_1.svg", format="svg")
    plt.show()
    run_experiments()
    
