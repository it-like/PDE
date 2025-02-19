import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (np.pi**2) * np.sin(np.pi * x)

def exact_solution(x):
    return np.sin(np.pi * x)

def fem_solver(m):
    a, b = 0.0, 1.0
    h = (b - a) / (m + 1)
    x = np.linspace(a, b, m+2)
    K_diff = (1/h) * np.array([[1, -1], [-1, 1]])
    A = np.zeros((m+2, m+2))
    F = np.zeros(m+2)
    for e in range(m+1):
        i = e
        j = e + 1
        A[i,i] += K_diff[0,0]
        A[i,j] += K_diff[0,1]
        A[j,i] += K_diff[1,0]
        A[j,j] += K_diff[1,1]

        F[i] += 0.5 * (x[j] - x[i]) * f(x[i])
        F[j] += 0.5 * (x[j] - x[i]) * f(x[j])
    A[0,:] = 0
    A[0,0] = 1
    A[-1,:] = 0
    A[-1,-1] = 1
    F[0] = 0
    F[-1] = 0
    U = np.linalg.solve(A, F)
    return x, U

if __name__ == "__main__":

    
    mesh_sizes = [2**i for i in range(6) if i >=1 ] # 1,2,3,4,5 for each: 2^each
    print(mesh_sizes)

    errors = []
    H = []
    
    for m in mesh_sizes:
        x, U = fem_solver(m)
    
        U_exact = exact_solution(x)
        # max norm between lines
        err = np.max(np.abs(U - U_exact))
        h = 1.0 / (m+1) # Step size
        
        errors.append(err)
        
        H.append(h) 
        
        print(f"m = {m}, h = {h:.2e}, error = {err:.2e}") # plots latest occurance of error by [-1]


    plt.figure()
    plt.grid(True)
    plt.loglog(H, errors, 'o-', label='FEM error')
    
    C = errors[0] / (H[0]**2)
    plt.loglog(H, [C * (h**2) for h in H], 'r--', label=r"~ $Ch^2$")

    plt.xlabel('h')
    plt.ylabel('Error')
    plt.legend()
    plt.title(r"Convergence of 1D FEM for $-u''(x) = \pi^2 \sin(\pi x)$")
    plt.savefig("assignment4/images/cg1_error.svg", format="svg")
    plt.show()
