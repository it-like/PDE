import matplotlib.pyplot as plt
import numpy as np

polynom = np.poly([1,-1])
x_vals = np.linspace(-2,4,1000)
def test_function(x):
    return x * np.exp(x) / (x + 1)**2

def create_midpoints(bounds: list, intervals: int):
    h = (bounds[1] - bounds[0]) / intervals
    sum_result = 0
    step = bounds[0]
    for _ in range(intervals):
        sum_result += test_function((step + h/2))
        step += h
    return sum_result * h

real = (np.e - 2)/2
approx = create_midpoints([0,1],100)
print(np.format_float_scientific(real-approx,5))
