import numpy as np
import matplotlib.pyplot as plt

def bvp_fd(a, b, N, delta, gamma, f_fun, ua, ub):
    """
    Solve the two-point BVP:
      -u''(x) + delta*u'(x) + gamma*u(x) = f(x) for x in (a,b)
    with u(a)=ua, u(b)=ub,
    using finite differences on a uniform grid with N interior points (+2 from edges)
    """
    h = (b - a) / (N + 1)
    
    x_full = np.linspace(a, b, N + 2)
    x_interior = x_full[1:-1]
    
    #   (-1 - h/2)*u_{i-1} + (2 + gamma*h^2)*u_i + (-1 + h/2)*u_{i+1} = h^2 * f(x_i)

    #    lower diagonal: delta * (-1 - h/2)
    #    centr diagonal:  2 + gamma * h^2
    #    upper diagonal: -1 + h/2
    
    lower_diag = np.full(N - 1,delta * (-1 - h/2))
    
    main_diag  = np.full(N, 2 + gamma * h**2)
    upper_diag = np.full(N - 1, -1 + h/2)
    
    # Avengers, assemble (k is if sub or super diag)
    A = np.diag(main_diag) + np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    # Right-hand side vector: f_i = h^2 * f(x_i)
    b_vec = h**2 * f_fun(x_interior)
    
    # Adjust for boundary conditions:
    b_vec[0] += (1 + h/2) * ua
    b_vec[-1] += (1 - h/2) * ub
    
    u_interior = np.linalg.solve(A, b_vec)

    # Put solution together    
    u_fd = np.concatenate(([ua], u_interior, [ub]))
    
    return x_full, u_fd
if __name__ == "__main__":

    a = 0.0
    b = 1.0

    N = 48         # interior nodes
    delta = 1.0    # coefficient on u'(x)
    gamma = 1.0    # coefficient on u(x)

    eps_stud = 17e-3

    # Boundary conditions
    ua = 0.0
    ub = np.sin(1) + 1 + eps_stud
    fine_x = np.arange(a,b,step=0.001)

    # From assignment
    def f_fun(x):
        return 2*np.sin(x) + np.cos(x) + x**2 + (2 + eps_stud)*x + eps_stud - 2

    # Exact solution
    def u_exact(x):
        return np.sin(x) + x**2 + eps_stud * x

    # Solve the BVP using the FD method
    x_vals, u_fd = bvp_fd(a, b, N, delta, gamma, f_fun, ua, ub)

    u_ex = u_exact(fine_x)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, u_fd, 'bo-', label='FD solution')
    plt.plot(fine_x, u_ex, 'r--', label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Finite Difference Approximation vs Exact Solution')
    plt.legend()
    plt.grid(True)
    #plt.show()
    svg_filename = 'assignment7/last_assignment.svg'
    plt.savefig(svg_filename, format="svg")
    print(f"Figure saved to '{svg_filename}'")
