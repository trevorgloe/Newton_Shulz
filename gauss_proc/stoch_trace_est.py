import numpy as np
from numpy.typing import NDArray
from collections.abs import Callable

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
        # print(z)
        # tot += z.T@A@z
        tot += np.dot(z, A@z)
        # print(tot)
    
    # print(tot)
    # print(n/l)
    return n *tot / l

# runs the Lanczos algorithm for computing the eigenvalues of A
# returns T_m, to be used in the stochastic Lanczos quadrature method
# this implimentation essentially copied from "Numerical Linear Algebra with Julia" - Eric Darve and Mary Wootters
def Lanczos(A:NDArray[np.float64], v:NDArray[np.float64], m):
    T = np.zeros((m, m))
    r = np.copy(v)
    beta = np.linalg.norm(r)
    q1 = np.zeros(v.shape)

    for k in range(m):
        q0 = np.copy(q1)
        q1 = r / beta
        r = A@q1
        alpha = np.dot(q1, r)
        T[k,k] = alpha
        if k > 1:
            T[k-1, k] = beta
            T[k,k-1] = beta

        r = r - alpha*q1 - beta*q0
        beta = np.linalg.norm(r)

    return T

# estimates v^T A v via the gaussian quadrature method produced from the stochastic lanczos quadrature
# estimates the quadratic form for a single vector
def LancQuadSingle(A:NDArray[np.float64], v:NDArray[np.float64], f:Callable, m:int):
    # assumes v is already normalized
    T = Lanczos(A, v, m+1)
    eval, evecs = np.linalg.eig(T)


# Estimates tr(f(A)) using the stochastic Lanczos quadrature method. Returns a single number. l is the number of vectors in the outter sum, m is the number of quadrature points used in each inner sum
# Algorithm taken from Ubaru, et. al. Fast estimation of tr(f(A)) via stochastic Lanczos quadrature (2017)
def StochLancQuad(A:NDArray[np.float64]):
    return 0
