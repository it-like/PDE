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
    print((sum_result * h) / 2)  

print(polynom)
print((np.e - 2)/2)
create_trapezoid([0,1],10000)
#plt.grid(True)
#plt.plot(x_vals, polynom(x_vals))
#plt.axis([-4,4,-4,4])
#plt.show()