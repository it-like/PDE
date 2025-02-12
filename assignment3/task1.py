#Task 1
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



# task a)
epsSTUD = 17e-3
lambd = 0.1 + epsSTUD # This is k parameter
P_0 = 5

def exact_sol(x):
    return P_0 * np.exp(lambd * x)

def f_x_der(y_values): # Euler f(y(t))~~ approx
    return (lambd * y_values)


x_vals = np.arange(0, 20, 0.0001)
y_vals = exact_sol(x_vals)

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel('t')
ax.set_ylabel('Population size at t')
ax.plot(x_vals, y_vals, label=r'$P(t)$')
ax.legend()
#plt.savefig("assignment3/images/P_t.svg", format="svg")
H = [.5, .25, .1]
T = 20


# task b)
for k in H:
    x_values = np.arange(0, T + k, k)
    y_values = [P_0]
    for t in x_values[1:]:
        y_values.append(y_values[-1] + k *f_x_der(y_values[-1]))
    ax.plot(x_values, y_values, label=f'Euler approx k={k}')

ax.axis([0, 20, 0, 50])
ax.legend()

axins = inset_axes(ax, width="30%", height="30%", loc='lower right')
axins.plot(x_vals, y_vals)
for k in H:
    x_values = np.arange(0, T + k, k)
    y_values = [P_0]
    for t in x_values[1:]:
        y_values.append(y_values[-1] + k * f_x_der(y_values[-1]))
    axins.plot(x_values, y_values)
axins.axis([17.5, 20, 40, 50])
axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)

mark_inset(ax, axins, loc1=2, loc2=1, ec="0.5")

plt.savefig("assignment3/images/euler_approx.svg", format="svg")
#plt.show()
plt.close()


#task c)

T = 1
y_vals_to_compare_at_t1 = []

for k in H:
    x_values = np.arange(0, T + k, k)
    y_values = [P_0]
    for t in x_values[1:]:
        y_values.append(y_values[-1] + k * f_x_der(y_values[-1]))
    y_vals_to_compare_at_t1.append(y_values[-1])

i=0

for approx in y_vals_to_compare_at_t1:
    print(f'Error is {np.format_float_scientific(abs(exact_sol(1) - approx),3)} against step size {np.format_float_scientific(H[i],5)}')
    i+=1
