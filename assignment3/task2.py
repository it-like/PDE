#Task 2
import matplotlib.pyplot as plt
import numpy as np

epsSTUD = 17e-3

H = 500
Z = 10
R = 0

#b)
λ = 0.005*epsSTUD
β = 0.01
ζ = 0.02

R_0 = 0
h = [.65, .1, .001]
def dH(t): return -β * H*Z
def dZ(t): return β*H*Z + ζ*R - λ*H*Z
def dR(t): return λ*H*Z - ζ*R

def system_next(t, h):
    global H
    global Z
    global R

    H = dH(t)*h + H
    Z = dZ(t)*h + Z
    R = dR(t)*h + R

for _h in h:
    H = 500
    Z = 10
    R = 0

    tvals = []
    hvals = []
    zvals = []
    rvals = []

    n = 10 / _h
    for i in range(int(n)):
        system_next(i*_h, _h)
        tvals.append(i*_h)
        hvals.append(H)
        zvals.append(Z)
        rvals.append(R)

    plt.plot(tvals, hvals, c='red')
    plt.plot(tvals, zvals, c='green')
    plt.plot(tvals, rvals, c='blue')
    plt.show()