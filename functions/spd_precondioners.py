"""
    Preconditioners to symmetric positive-definite (SPD) matrices. These are mainly for use in the Gaussian process code, as the kernel matrix will be SPD
"""

import numpy as np
import functions.preconditioners as pre

"""
    randomly pivoted cholseky (RPC) computes the partial cholesky factorization (nystrom approximation) of given matrix, using the random pivoting strategy
    each diagonal pivot is chosen with probability A_{ii}/tr(A)
    To avoid passing around the full matrix, too much, this function just returns the pivot selections, I
"""
def RandomlyPivotedCholesky(A : np.ndarray, kmax : int):
    I = [] # index set
    n = A.shape[0]
    temp = np.copy(A)
    d = np.diag(A)
    for k in range(kmax):
        p = abs(d)
        p = p / sum(p)
        newi = np.random.choice(n, size=1, p=p) # sample pivot from diagonal
        I.append(newi[0])
        # print(temp[:,newi])
        # print(temp[newi,:])
        temp = temp - (1/temp[newi, newi])*temp[:,newi]@temp[newi, :]
        d = np.diag(temp)

    return I

"""
    Ridge with partial cholesky factorization. Solve the system (mu*I + A)x = b where A is given as a low rank matrix, coming from a cholseky factorization. This uses the Woodbury matrix identity to solve the system. Write A=LU^{-1}L^T where L is a subset of A's columns, and U is the principal submatrix with diagonals from I
    Done with the following algorithm:
    1) b1 = b/mu
    2) bhat = L^T*b
    3) solve (U + L^T*L/mu)x = bhat (use cholseky or something
    4) b2 = L*x/mu^2
    5) x = b1 + b2
"""
def RidgewPartialChol(mu : np.float64, A : np.ndarray, I, b : np.ndarray):
    L = A[:, I]
    U = A[np.ix_(I,I)]
    print(U.shape)
    b1 = b / mu
    bhat = L.T@b
    # print(L.shape)
    # print(L.T@L)
    x = np.linalg.solve(U + L.T@L/mu, bhat)
    b2 = L@x / (mu**2)
    return b1 - b2

"""
    Vecchia approximation via the conventional algorithm. Taken from Webber et. al, "Everything is Vecchia"
    Computes a factored inverse cholesky approximation to A, of the form A=C^-1 D C^-* where the non-zero off-digonal indices of C are given by S[i]. 
    A is a positive definition matrix, and S[i] is a list of ints describing the non-zero off-diagonal elements of C
"""
def Vecchia(A : np.ndarray, S : list):
    n = A.shape[0]
    C = np.eye(n)
    D = np.zeros((n,n))
    I = np.eye(n)
    for i in range(n):
        p = np.hstack([I[:,S[i]], I[:,[i]]])
        temp_p = A@p
        bigM = p.T @ temp_p
        print(bigM.shape)
        M = bigM[:len(S[i]), :len(S[i])]
        print(M)
        # v = np.squeeze(bigM[:, len(S[i])])
        v = bigM[:len(S[i]), len(S[i])]
        print(v)
        Ml = np.linalg.cholesky(M)
        temp = pre.back_substitution(Ml.T, -v)
        x = pre.forward_substitution(Ml, temp)
        # print(x)
        C[i,S[i]] = x.T
        D[i,i] = bigM[len(S[i]), len(S[i])] + np.dot(x, v)

    return C, D



