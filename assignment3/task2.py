import matplotlib.pyplot as plt
import numpy as np

eps_stud = 17e-3
lam = 0.005 * eps_stud
beta = 0.01
zeta = 0.02

def system_deriv(state, t):
    H, Z, R = state
    dH = -beta * H * Z
    dZ = beta * H * Z + zeta * R - lam * H * Z
    dR = lam * H * Z - zeta * R
    return np.array([dH, dZ, dR])
step_sizes = [0.65, 0.1]

for dt in step_sizes:
    t_max = 10
    n_steps = int(t_max / dt) # float for dt = 0.65
    tvals = np.linspace(0, t_max, n_steps + 1)
    state = np.array([500, 5, 0]) # initial values 500 humans, 5 zombies
    
    Hvals, Zvals, Rvals = [state[0]], [state[1]], [state[2]]
    
    for i in range(n_steps):
        state = state + dt * system_deriv(state, tvals[i])
        Hvals.append(state[0])
        Zvals.append(state[1])
        Rvals.append(state[2])
        print(state[0])
        print(state[1])
        print(state[2])
    plt.figure(figsize=[12,8])
    plt.grid(True)
    plt.plot(tvals, Hvals, c='green', label='Humans')
    plt.plot(tvals, Zvals, c='red', label='Zombies')
    plt.plot(tvals, Rvals, c='grey', label='Removed')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Euler scheme with h = {dt}')
    plt.legend()
    plt.show()
