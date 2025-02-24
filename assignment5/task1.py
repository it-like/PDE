import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

epsSTUD = 17e-3 # Gustav Gille

# Time partition
T = 1.0           
n_time = 100      
k = T / n_time      # Time discretisation

# Space partition
m = 10
N_nodes = m + 2
h = 1.0 / (m +1)    # Space discretisation
  
# this is m+1 subintervals, +1 for connecting lines (1 line needs two edges)
X = np.linspace(0, 1, m + 2) 
N = len(X) # All points/indexes

def assemble_matrices(m, h):
    # Global mass and stiff matrices
    M = np.zeros((N, N))
    S = np.zeros((N, N))

     
    for e in range(N-1):
        indices = [e, e + 1] # Local left and right hat functions
        
        M_e = (h/6) * np.array([[2, 1],
                                [1, 2]])
        
        S_e = (1/h) * np.array([[1, -1],
                                [-1, 1]])
        
        for i in range(2):
            for j in range(2):
                M[indices[i], indices[j]] += M_e[i, j]
                S[indices[i], indices[j]] += S_e[i, j]

    # Exclude at x = 1 by Dirichlet condition, zero out row
    # add 1 on diagonal to avoid singularity (as before)
    M[m+1:] = 0
    M[m+1, m+1] = 1
    S[m+1:] = 0
    S[m+1, m+1] = 1

    return M, S



def f(x, t):
    #u = - t(epsSTUD/2)(x^2-1)
    #u_t = - (epsSTUD/2)*(x**2-1)
    #u_xx = -epsSTUD*t
    return - (epsSTUD/2) * (x**2 - 1) + epsSTUD * t



def assemble_f(x, t):
    F = np.zeros(N)
    # We will have to approximate f by simpsons,
    # that means we have to apply it to 
    # both the 'left' and 'right' hat functions
    for e in range(N - 1):
        a = x[e]
        b = x[e+1]
        midpoint = (a + b) / 2.
        
        h_elem = b - a  
        
        # Left local basis function on [a,b]: φ(a)=1, φ(b)=0. 
        phi_left_a = 1.0
        phi_left_m = 1/2
        I_left = (h_elem/6) * (phi_left_a * f(a, t) 
                               + 4 * phi_left_m * f(midpoint, t) )                               
        F[e] += I_left

        # Right local basis function: φ(a)=0, φ(b)=1.   
        phi_right_m = 1/2
        phi_right_b = 1.0
        I_right = (h_elem/6) * (+ 4 * phi_right_m * f(midpoint, t) 
                                + phi_right_b * f(b, t))
        F[e+1] += I_right
    
    # Enforce the Dirichlet condition at the last node: u(1)=0.
    F[m+1] = 0
    return F

mass, stiff = assemble_matrices(m, h)



from numpy.linalg import solve
xi = np.zeros(m+2) 

sol_time = [xi.copy()]
time_vec = [0.0]
'''Backward Euler scheme'''
# Rearranged approximation, equation 12
for l in range(1, n_time+1): # start at 1, 0 already done from above
    t_l = l * k
    F_l = assemble_f(X, t_l)
    lhs_matrix = mass + k * stiff  
    rhs = mass @ xi + k * F_l

    xi = solve(lhs_matrix, rhs)
    sol_time.append(xi.copy())
    time_vec.append(t_l)
sol_time = np.array(sol_time)  
time_vec = np.array(time_vec)
 
def u_exact(x, t):
    # u_t = -(epsSTUD*x**2)/2 + epsSTUD/2
    # u_xx = -epsSTUD*t
    return - (epsSTUD/2) * t * (x**2 - 1)


'''See for 5 different times'''
plt.figure(figsize=(10,8))

plot_times = [0*T/4, 1*T/4, 2*T/4, 3*T/4, 4*T/4] # Time disc. intervals
colors = ['r', 'g', 'b', 'm', 'c']

for pt, col in zip(plot_times, colors):
    # Find nearest time step index from solutions
    idx = np.argmin(np.abs(time_vec - pt))
    xi_FE = sol_time[idx, :]
    x_fine = np.linspace(0, 1, 2000)
    u_ex = u_exact(x_fine, time_vec[idx])
    
    plt.subplot(2,3,colors.index(col)+1)
    plt.plot(x_fine, u_ex, 'k-', label='Exact')
    plt.plot(X, xi_FE, col+'o--', label='FE')
    plt.axis([0,1,0,0.01])
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f"t = {time_vec[idx]}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
plt.suptitle("Comparison of FE and Exact Solutions", y=1)
plt.savefig("assignment5/images/plot_over_5_times.svg", format="svg")
plt.close()


'''Compute 3d plot of x,t mesh for z=u(x,t)'''
X, T = np.meshgrid(X, time_vec)  
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot( projection='3d')
#Python>matlab
surf = ax.plot_surface(X, T, sol_time, cmap='viridis', edgecolor='none')
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x,t)')
ax.set_title('PDE solution evolution over time')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("assignment5/images/3d.svg", format="svg")
plt.close()


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot( projection='3d')
surf = ax.plot_surface(X, T, sol_time, cmap='viridis', edgecolor='none')
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('u(x,t)')
ax.set_title('PDE solution evolution over time')
fig.colorbar(surf, shrink=0.5, aspect=5)

def animate(angle):
    ax.view_init(elev=30, azim=angle)
    return fig,

rot_animation = animation.FuncAnimation(fig, animate, frames=np.linspace(0, 360, 120), interval=60)
rot_animation.save('assignment5/images/rotating_plot.gif', writer='imagemagick', fps=30)
plt.close()
