#task3.py

import matplotlib.pyplot as plt
import numpy as np

polynom = np.poly([1,-1])
x_vals = np.linspace(-2,4,1000)
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
plt.title('Trapezoid Rule')
plt.plot(x, y, c="k")
plt.show()



def create_midpoints(bounds: list, intervals: int):
    h = (bounds[1] - bounds[0]) / intervals
    sum_result = 0
    step = bounds[0]
    for _ in range(intervals):
        sum_result += test_function((step + h/2))
        step += h
    return sum_result * h

x,y = [], []

true_ans = (np.e - 2)/2

for i in range(8):
    approx = create_midpoints([0,1], 2**i)
    error = np.abs(approx - true_ans)
    x.append(i)
    y.append(error)

plt.grid(True)
plt.title('Midpoint Rule')
plt.plot(x, y, c="k")
plt.show()
