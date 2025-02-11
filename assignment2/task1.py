import matplotlib.pyplot as plt
import numpy as np

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

real = (np.e - 2)/2
approx = create_trapezoid([0,1],100)
print(np.format_float_scientific(real-approx,5))