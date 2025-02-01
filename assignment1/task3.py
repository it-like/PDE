#task3.py
import numpy as np
import matplotlib.pyplot as plt
from task2 import *

''' 
########################implementation########################

Lagrange polynomials fits a polynomial to a set of points by creating
multiple equations which equal zero for the point not of interest.

Say we have three points at [1,1], [2,0], [3,4]. Then the following 
expression can be used to build a foundation to solve the problem
    
    (x-1)(x-2)(x-3).

If the point [1,1] is to be examined, then we remove the term corresponding
to zeroing it out, that is the (x-1) part.
This point's y-value will be (1-2)(1-3) = (-1)(-2) = 2 at the point, 
which is not correct (should be 1). 
A solution is to scale all points to one and then rescale with another
function. The downscaling can be used by normalising 
    (x-2)(x-3)/         # (x-1) removed
    (1-2)(1-3)          # (x-1)
    = 1

For point [2,0] this would correspond to 
    (2-1)(2-3)/         #(x-2) 
    (2-1)(2-3)          #(x-2) 
    = 1

This indicator function is known as the Lagrange polynomials, presented as
l_i = \Pi_{j = 0,j \neq i}^n\frac{x-x_j}{x_i-x_j}
where i represents the current point and j all other points.
'''
def lagrange_lambdas(x_vals):
    list_of_pols = []    
    n = len(x_vals)
    for i in range(n):
        p = np.poly1d([1.]) # New for each l_i
        for j in range(n):
            if i != j:
                p *= np.poly1d([1, -x_vals[j]]) / (x_vals[i] - x_vals[j])
        list_of_pols.append(p)
    return list_of_pols

'''
The helper function scaling the points appropriately can be introduced as 

p(x) = \sum^n_{i=0}y_il_i(x),
where y_i is the corresponding value for the the point i.

This can be implemented as the following
'''
def pi_interpolation(list_of_pols, cor_y_vals):
    answer = 0
    for polynom, y_value in zip(list_of_pols, cor_y_vals):
        print(answer)
        answer += polynom*y_value
    return answer



x_vals = [1,2,3]
y_vals = [1,4,2]
np_poly_coeffs = np.polyfit(x_vals, y_vals, 2)
np_poly = np.poly1d(np_poly_coeffs)
all_polys = lagrange_lambdas([1,2,3])


'''individual L_i plots'''

# Implemented interpolation
ans = pi_interpolation(all_polys, y_vals)

'''
#####COMPUTE ERROR BETWEEN SELF_IMPLEMENTED AND LIBRARY#####
From the lectures the error is bounded by some different norms, but
can also directly be computed by estimating the norm numerically
''' 
def compute_error_between_pols(pol_1, pol_2, interval, n=1000):
    a, b = interval
    dx = (b - a) / n
    x_mid = np.linspace(a + dx/2, b - dx/2, n) # dx steps n amount of times
    diff = pol_1(x_mid) - pol_2(x_mid)
    return np.sqrt(dx * np.sum(diff**2)) # sums ups the squared error error

error = compute_error_between_pols(ans,np_poly, [1,3])


lin_x_vals = np.linspace(0,4,1000)
plt.axis([0.5,3.5,-1,4.5])
for i, poly in enumerate(all_polys):
    plt.plot(lin_x_vals, poly(lin_x_vals), label=f'$l_{i}(x)$')
plt.grid(True)
plt.scatter(x_vals, y_vals, marker='.',c='black',s=120) # data points
plt.scatter(x_vals,np.zeros_like(x_vals),marker='.', c='black',s=120) # data points with y=0
plt.axhline(0, color='black') # show y = 0 line
plt.plot(lin_x_vals, ans(lin_x_vals), label=f'$Predicted$',linestyle='-', c='grey')
plt.plot(lin_x_vals, np_poly(lin_x_vals), label=f'$True$',linestyle='dashed',c='black')
plt.legend()
plt.title(f'Error between estimated and true is {error:.2e}.')
plt.savefig('assignment1/photos/task3.png')


