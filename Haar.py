import numpy as np
import numpy.random as r

"""
    Givens rotation of matrix A for angle c = cos(theta) at index i,j
    Does not actually create the Givens matrix, just computes the rotation of each column
    c is the cosine of theta for the rotation
"""
def givensRot(A: np.ndarray, c: float, i: int, j: int):
    # loop through each column and rotate it according to the givens rotation
    out = A.copy()
    for l in range(A.shape[1]):
        out[:,l] = givensRotVec(A[:,l], c, i, j)

    return out

"""
    Givens rotation for a vector, just here as a helper function for givensRot
    c is the cosine of the angle
"""
def givensRotVec(v: np.ndarray, c: float, i: int, j: int):
    newv = v.copy()
    s = np.sqrt(1 - c**2)
    print(s)
    print(newv[i])
    print(newv[j])
    newv[i] = newv[i]*c - newv[j]*s
    newv[j] = newv[i]*s + newv[j]*c

    return newv

"""
    Create a random nxn orthogonal matrix according to the Haar measure
    Done via the algorithm described in "Generation of Random Orthogonal Matrices", Anderson, et. al. 1985
"""
def randOrth(n: int):
    # start be initializing a D matrix of +/- 1
    D = np.diag(1 - 2*(r.binomial(n=1, p=0.5, size=n)))
    G = D.copy()
    print(G)

    for i in range(n-1, 0, -1):
        print(f"i = {i}")
        # for each i, we will create n-i givens rotations
        # create chi-squared variables to generate the beta distributed random variables
        x = r.chisquare(df = 1)
        s = x
        for j in range(n, i, -1):
            # create the next beta distributed number via
            x = r.chisquare(df=1)
            y = s / (s + x)
            s = s + x
            print(f"Cosine is {y}")
            print(f"Index is {i}, {j}")
            # now y = cos^2 so rotate by cos
            G = givensRot(G, np.sqrt(y), j-1, i-1)
            print(G)

    return G




