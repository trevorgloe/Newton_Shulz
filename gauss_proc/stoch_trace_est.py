import numpy as np
from numpy.typing import NDArray

# returns a rademacher distributed random vector
def rademacher(n:int):
    z = np.random.choice([-1,1], size=n)
    return z / np.linalg.norm(z)
# estimates the trace of a given matrix via a stochastic trace estimator using rademacher distributed random vectors
# Assumes A is a square matrix 
def STE(A:NDArray[np.float64], l:int):
    n = A.shape[0] # should be square
    tot = 0.0
    for i in range(l):
        z = rademacher(n)
        print(z)
        tot += z.T@A@z
    
    print(tot)
    print(n/l)
    return 1 / l * tot

