import numpy as np
import numpy.random as r
import numpy.typing as npt

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
    v = v.astype(np.float64)
    newv = v.copy()
    s = np.sqrt(1 - c**2)
    newv[i] = v[i]*c - v[j]*s
    newv[j] = v[i]*s + v[j]*c

    return newv

# """
#     Create a random nxn orthogonal matrix according to the Haar measure
#     Done via the algorithm described in "Generation of Random Orthogonal Matrices", Anderson, et. al. 1985
# """
# def randOrth(n: int):
#     # start be initializing a D matrix of +/- 1
#     D = np.diag(1 - 2*(r.binomial(n=1, p=0.5, size=n))).astype(np.float64)
#     G = D.copy()
#     # print(G)
#
#     for i in range(n-1, 0, -1):
#         # print(f"i = {i}")
#         # for each i, we will create n-i givens rotations
#         # create chi-squared variables to generate the beta distributed random variables
#         x = r.chisquare(df = 1)
#         s = x
#         for j in range(n, i, -1):
#             # create the next beta distributed number via
#             x = r.chisquare(df=1)
#             y = s / (s + x)
#             s = s + x
#             print(f"Cosine is {y}")
#             print(f"Index is {i}, {j}")
#             # now y = cos^2 so rotate by cos
#             G = givensRot(G, np.sqrt(y), j-1, i-1)
#             # print(G)
#
#     return G

"""
    Create a random nxn orthogonal matrix according to the Haar measure
    Done via the algorithm of "The Efficient Generation of Random Orthogonal Matrices with an Application to Condition Estimators", G.W. Stewart 1980
"""
def randOrth(n: int):
    # first generate n normally distributed random vectors
    all_vecs = np.zeros((n,n))
    for i in range(n):
        # print(i)
        all_vecs[i:n,i] = r.randn(n-i)
   
    # print(all_vecs)
    # create the diagonal matrix of sgn(x[1]) for each x in the vectors generate
    d = [np.sign(all_vecs[i,i]) for i in range(n)]
    D = np.diag(d)
    G = np.eye(n) # start with identity matrix
    # print(D)

    for i in range(n-1):
        # start with the n-i+1 length vector, and create the householder reflection that transforms that vector to ||x|| e_1
        x = all_vecs[i:n,i]
        # print(x.shape)
        # e = np.zeros(x.shape)
        # e[0] = np.sign(x[n-i-1])*1.0
        # v = x - np.linalg.norm(x)*e
        v = x
        v[0] = v[0] + np.sign(x[0])*np.linalg.norm(x)
        # print(v)
        beta = 2 / np.dot(v,v)

        # H = np.eye(n-i) - beta * np.outer(v,v) 
         
        # D[i:n,i:n] = D[i:n,i:n] - beta*np.outer((D[i:n,i:n]@v), v.T)
        # G[i:n,i:n] = G[i:n,i:n] @ H
        # G = G@np.block([[np.eye(i), np.zeros((n-i,i))]
        # G[:,i:n] = G[:,i:n]@H # transform G by [I 0 ; 0 H]
        G[:, i:n] = G[:,i:n] - beta * np.outer((G[:,i:n]@v), v) # transform G by [I 0 ; 0 H]
        # print(G)
        # print(G@G.T)

    # fix the signs of the rows: flip the sign of the ith row if the ith element of d is -1
    G = D @ G
    return G
