import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (np.pi**2) * np.sin(np.pi * x)

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

def compute_error_on_fine_grid(x_coarse, U_coarse, exact_fun, num_fine_points=2001):
    x_fine = np.linspace(x_coarse[0], x_coarse[-1], num_fine_points)
    U_fine = np.interp(x_fine, x_coarse, U_coarse)
    U_exact_fine = exact_fun(x_fine)
    diff = U_fine - U_exact_fine
    L2_error = np.sqrt(np.trapz(diff**2, x_fine))
    return L2_error

def main():
    exact_sol = lambda xx: np.sin(np.pi * xx)
    L_values = [1, 2, 3, 4, 5]
    hs = []
    errors = []
    for L in L_values:
        m = 2**L
        x_coarse, U_coarse = fem_solver(m)
        e = compute_error_on_fine_grid(x_coarse, U_coarse, exact_sol)
        h = 1.0 / (m + 1)
        hs.append(h)
        errors.append(e)
        print(m)
    
    plt.loglog(hs, errors, 'o--')
    plt.xlabel("h")
    plt.ylabel("Error")
    plt.grid(True)
    p_coeff = np.polyfit(np.log(hs), np.log(errors), 1)
    p_est = p_coeff[0]
    plt.text(hs[len(hs)//2], errors[len(hs)//2], f"Order ~ {p_est:.2f}", fontsize=12, color='red')
    plt.show()


    

if __name__ == "__main__":
    m = 1100
    broken_for_exact = np.arange(0,1,0.00001)
    x, U = fem_solver(m)
    u_exact = np.sin(np.pi * broken_for_exact)  # exact solution
    
    plt.figure(figsize=(12,9))
    plt.plot(broken_for_exact, u_exact, 'k-', label='Exact $u(x)=\sin(\pi x)$')
    plt.plot(x, U, 'ro--', label='FEM cG(1) solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.title('FEM cG(1) Approximation for 1D Poisson Problem')
    plt.show()
    main()
    
