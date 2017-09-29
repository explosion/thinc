# This executes the code in the module, initialising blis
import blis.blis
cimport numpy as np

def variance(float[::1] X):
    return Vec.variance(&X[0], X.shape[0])

