## Conjugate gradient method
# This file contains the conjugate gradient algorithm
import numpy as np
from numpy.typing import NDArray

# Conjugate Gradient method
# A is assumed to be a SYMETRIC nxn matrix
# b is assumed to be a n-dimensional vector
# roughly follows the formulation by Eric Darve and Mary Wooters - Numerical Linear Algebra with Julia
def ConjugateGradient(A:NDArray[np.float64], b:NDArray[np.float64], verbose=False, maxk=None, tol=None, return_all_res=False):
    n = A.shape[0]
    if maxk is None:
        maxk = n # default value
    if tol is None:
        tol = 1e-16 * n #default tolerance, numerical precision
    if verbose:
        print(f"Running conjugate gradient method for {n}x{n} matrix")
    
    all_res = []
    x = np.zeros(n)
    r = np.copy(b)
    p = np.zeros(n)
    rho = 1
    for k in range(maxk):
        rho_old = rho
        rho = np.dot(r,r)
        old_p = np.copy(p)
        p = r + (rho / rho_old) * p
        if verbose:
            print(rho / rho_old)
            print(np.dot(old_p, p))

        q = A@p
        mu = rho / np.dot(p, q)
        x += mu * p
        r -= mu * q
        res = np.linalg.norm(r)
        if verbose:
            print(f"Iteration {k+1}, res={res}")
        if return_all_res:
            all_res.append(res)

        if res < tol:
            break

    if return_all_res:
        return x, all_res
    else:
        return x

