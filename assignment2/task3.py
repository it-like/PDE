#task3.py

import matplotlib.pyplot as plt
import numpy as np
C = 50
def test_function(x):
    return x * np.exp(x) / (x + 1)**2

def create_trapezoid(bounds: list, intervals: int):
    h = (bounds[1] - bounds[0]) / intervals
    sum_result = 0
    step = bounds[0]
    for _ in range(intervals):
        sum_result += test_function(step) + test_function(step + h)
        step += h
    return (sum_result * h) / 2

x,y = [], []

true_ans = (np.e - 2)/2

for i in range(8):
    approx = create_trapezoid([0,1], 2**i)
    error = np.abs(approx - true_ans)
    x.append(i)
    y.append(error)

plt.grid(True)
n_2 = lower_bound = [(1/2**i)/C for i in range(8)]
#plt.plot(x, n_2, label=rf'$\frac{{1}}{{n^2\, {C}}}$', color='green')

plt.title('Trapezoid Rule')
plt.plot(x, y,'--', label='Trapezoid',color='black')
#plt.legend()
#plt.savefig("assignment2/images/Trap.svg", format="svg")
#plt.close()




def create_midpoints(bounds: list, intervals: int):
    h = (bounds[1] - bounds[0]) / intervals
    sum_result = 0
    step = bounds[0]
    for _ in range(intervals):
        sum_result += test_function((step + h/2))
        step += h
    return sum_result * h

x,y = [], []
for i in range(8):
    approx = create_midpoints([0,1], 2**i)
    error = np.abs(approx - true_ans)
    x.append(i)
    y.append(error)

plt.grid(True)
plt.title('Midpoint Rule')
plt.plot(x, n_2, label=rf'$\frac{{1}}{{n^2\, {C}}}$', color='green')
plt.plot(x, y,'-', label='Midpoint',color='black')
plt.legend()
plt.savefig("assignment2/images/both.svg", format="svg")
