import numpy as np

def orifunc(x):
    return 1/(1+np.exp(-x))

def d_func(x):
    y=orifunc(x)
    return y*(1-y)
