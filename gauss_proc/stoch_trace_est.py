import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable

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
    n = A.shape[0]
    T = np.zeros((m, m))
    Q = np.zeros((n, m))
    r = np.copy(v)
    # print(r)
    # q1 = np.copy(v)
    beta = np.linalg.norm(r)
    # Q[:,0] = r / beta

    for k in range(m):
        q1 = np.copy(r)
        q1 = q1 / beta
        Q[:,k] = q1
        # print(Q)

        if k>0:
            T[k-1,k] = beta
            T[k,k-1] = beta

        r = A@q1
        alpha = np.dot(q1, r)
        T[k,k] = alpha
        for j in range(k+1):
            val = np.dot(r, Q[:,j])
            r = r - val * Q[:,j]

        beta = np.linalg.norm(r)

    return T, Q

# estimates v^T A v via the gaussian quadrature method produced from the stochastic lanczos quadrature
# estimates the quadratic form for a single vector
def LancQuadSingle(A:NDArray[np.float64], v:NDArray[np.float64], f:Callable, m:int):
    # assumes v is already normalized
    T, Q = Lanczos(A, v, m)
    # print(T)
    eval, evecs = np.linalg.eig(T)
    # print(eval)
    # each column of evecs is an eigenvalue, we want the first element of each of these 
    e1 = np.zeros(m)
    e1[1] = 1
    tau_big = e1.T@evecs
    tau = tau_big
    # print(tau)
    tau_sq = np.power(tau, 2)
    # print(tau_sq)
    theta = eval
    f_eval = [f(t) for t in theta]
    return np.dot(tau_sq, f_eval)


# Estimates tr(f(A)) using the stochastic Lanczos quadrature method. Returns a single number. l is the number of vectors in the outter sum, m is the number of quadrature points used in each inner sum
# Algorithm taken from Ubaru, et. al. Fast estimation of tr(f(A)) via stochastic Lanczos quadrature (2017)
def StochLancQuad(A:NDArray[np.float64], f:Callable, l:int, m:int):
    n = A.shape[0]
    tot = 0.0
    for i in range(l):
        v = rademacher(n)
        tot += LancQuadSingle(A, v, f, m)

    return (n / l) * tot

# Partial cholseky decomposition as a preconditioner to a kernel matrix
# matrix is assumed to be symmetric
def PartialCholseky(A:NDArray[np.float64], m:int):
    # dont do any pivoting for now, just use the first m elements for the partition
    n = A.shape[0]
    A11 = A[0:m,0:m]
    A22 = A[m:,m:]
    A12 = A[0:m, m:]
    A21 = np.copy(A12).T
    L11 = np.linalg.cholesky(A11)
    # print(L11.shape)
    D11 = np.diag(np.diag(L11))
    L11 = L11 @ np.linalg.inv(D11) # normalize L11 so its unit lower triangular
    # now technically, the diagonal piece of the cholseky decomp should be D^2
    D11 = D11@D11
    # print(A21.shape)
    L21 = A21 @ np.linalg.inv(L11.T) @ np.linalg.inv(D11)
    D22 = np.diag(np.diag(A22) - np.diag(L21@D11@L21.T))
    L = np.block([[L11, np.zeros((m, n-m))], [L21, np.eye(n-m)]])
    D = np.block([[D11, np.zeros((m, n-m))], [np.zeros((n-m, m)), D22]])
    return L, D

