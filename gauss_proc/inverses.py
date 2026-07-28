from numpy._core.numeric import ndarray
from abc import ABC, abstractmethod

import numpy as np

import sys
from pathlib import Path

# from functions.preconditioners import forward_substitution
REPO_ROOT = Path.cwd().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import forward_substitution, back_substitution


# abstract class from which the different methods can inherit
"""
    Every class inheriting from InverseMethod is designed to compute the log liklihood of the gaussian process defined by a given covariance matrix K, and noise level, sig
    vecSolve computes the solution to (sig^2*I + K)x = b
    sig and K are passed separately because some methods will focus on approximating K separately than the regularized matrix
"""
class InverseMethod(ABC):
    @abstractmethod # so implementation is required
    def vecSolve(self, K : np.ndarray, sig : float, b : np.ndarray) -> np.ndarray:
        pass

    def __init__(self, prints = False, cache = True):
        self.prints = prints
        self.cache = cache
        self.inv_computed = False # flag to check if the inverse has already been computed, that way its not computed multiple times

    def invalidateInverse(self):
        self.inv_computed = False

    def log(self, s): # prints out s if self.prints is true
        if self.prints:
            print(s)

# numpy inverse method
"""
    This class will compute the inverse via the standard numpy function, np.linalg.inv
    It computes the full inverse, caching it for further use. When invalidateInverse is called, it will be forced to compute the inverse again
"""
class NumpyInverse(InverseMethod):
    def __init__(self, prints = False, cache = True):
        super().__init__(prints, cache)
        self.Ki = None

    def computeInv(self, K : np.ndarray, sig : float):
        if (self.cache and (not self.inv_computed)):
            self.log("Computing inverse via numpy")
            self.Ki = np.linalg.inv(K + sig**2 * np.eye(K.shape[0]))
            self.inv_computed = True

    def vecSolve(self, K : np.ndarray, sig : float, b : np.ndarray):
        self.computeInv(K, sig)
        return self.Ki @ b


# Cholesky factorization
"""
    This class computes the inverse via the full cholesky factorization (unpivoted as of now, but will add pivoting in the future)
    It stores the cholesky factor L until invalidateInverse is called
    Backsubstitution is used to solve Lx = b for the vector solver
    The forward and back substitution algorithms are taken from functions/preconditioners
"""
class Cholesky(InverseMethod):
    def __init__(self, prints = False, cache = True):
        super().__init__(prints, cache)
        self.L = np.array([])

    def computeL(self, K : np.ndarray, sig : float):
        if self.cache and (not self.inv_computed):
            self.log("Computing cholesky factorization")
            self.L = np.linalg.cholesky(K + sig**2 * np.eye(K.shape[0])) # returns L such that L@L.T = K

    def vecSolve(self, K : np.ndarray, sig : float, b : np.ndarray):
        self.computeL(K, sig)
        # solve the system in 2 steps
        # 1) temp = L^{-1} * b (forward substitution)
        # 2) x = (L.T)^{-1} * temp (back substitution)
        temp = forward_substitution(self.L, b)
        self.log(f"forward sub has error {np.linalg.norm(self.L @ temp - b)}")
        x = back_substitution(self.L.T, temp)
        self.log(f"back sub has error {np.linalg.norm(self.L.T @ x - temp)}")
        return x

# Newton-Schulz
"""
    This class computes the inverse via the Newton-Schulz algorithm
    The class allows for configurable preconditioners to the algorithm, current allowed preconditioners are:
        - incomplete cholesky factorization
        - partial cholesky factorization via randomly pivoted cholesky algorithm
"""
class NewtonSchulz(InverseMethod):
    def __init__(self, prints = False, cache = True,
                 precond = "None", # preconditioner, None for no preconditioner, ichol for incomplte cholesky, pchol for partial cholesky
                 pc_rank = 1, # rank of partial cholesky, if used
                 ic_tol = 0.0 # drop tolerance of incomplete cholesky, if used
                 ):
        super().__init__(prints, cache)
        self.Khati = np.array([])
        self.precond = precond
        self.pc_rank = pc_rank
        self.ic_tol = ic_tol

    def computeInv(self, K : np.ndarray, sig : float):
        # check if inverse has already been computed
        if (not self.cache) or self.inv_computed:
            return
        
        # compute preconditioner 
        match self.precond:
            case "pchol":

