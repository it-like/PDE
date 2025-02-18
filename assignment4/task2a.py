import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (np.pi**2) * np.sin(np.pi * x)


def fem_poisson_1D_cg2(m):
    a = 0.0
    b = 1.0
    h = (b - a) / m       
    n_nodes = 2 * m + 1   

    x = np.linspace(a, b, n_nodes)

    A = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)


    K_local = (1.0/h) * (1.0/3) * np.array([[ 7, -8,  1],
                                             [-8, 16, -8],
                                             [ 1, -8,  7]])

    for e in range(m):
        
        i = 2*e
        j = 2*e + 1
        k = 2*e + 2

        
        x_left  = a + e * h        
        x_mid   = x_left + h/2     
        x_right = x_left + h       


        # Simposon rule
        F_local = (h/6  ) * np.array([
            f(x_left),
            4 * f(x_mid),
            f(x_right)
        ])
        
        A[i, i] += K_local[0, 0]
        A[i, j] += K_local[0, 1]
        A[i, k] += K_local[0, 2]

        A[j, i] += K_local[1, 0]
        A[j, j] += K_local[1, 1]
        A[j, k] += K_local[1, 2]

        A[k, i] += K_local[2, 0]
        A[k, j] += K_local[2, 1]
        A[k, k] += K_local[2, 2]


        F[i] += F_local[0]
        F[j] += F_local[1]
        F[k] += F_local[2]

    # Apply Dirichlet conditions
    A[0, :] = 0.0
    A[0, 0] = 1.0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    
    U = np.linalg.solve(A, F)

    return x, U

if __name__ == '__main__':
    m = 10
    broken_for_exact = np.arange(0,1,0.00001)
    x, U = fem_poisson_1D_cg2(m)
    u_exact = np.sin(np.pi * broken_for_exact)  # exact solution

    plt.figure(figsize=(12,9))
    plt.plot(broken_for_exact, u_exact, 'k-', label='Exact $u(x)=\sin(\pi x)$')
    plt.plot(x, U, 'ro--', label='FEM cG(2) solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.title('FEM cG(2) Approximation for 1D Poisson Problem')
    plt.savefig("assignment4/images/fem_cg2_approx.svg", format="svg")
    plt.show()
    
