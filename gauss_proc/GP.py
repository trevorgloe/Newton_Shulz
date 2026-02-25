import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from typing import Callable
from numpy.typing import NDArray
import math
import scipy.linalg as scil
from matrix_functions import Newton_Shulz

class Kernel:
    def __init__(self, fnc: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], float], n: int, fnc_prime: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], theta_n: int):
        # kernel function like k(x,y,theta) where theta is a vector of parameters to use
        self.fnc = fnc # kernel function, k
        self.fnc_p = fnc_prime # dk / dtheta
        self.n = n
        self.mat = np.zeros((n,n))
        self.theta = np.zeros((theta_n, 1)) # initialize all parameters to 0

    # set the parameters of the kernel function
    def setTheta(self, theta:NDArray[np.float64]):
        if theta.shape[0] != self.theta.shape[0]:
            raise ValueError("Dimension of input parameters does not match that of kernel function")

        self.theta = theta

    # compute the kernel matrix for a given set of n points
    def compute(self, x:NDArray[np.float64]):
        if x.shape[1] != self.n:
            raise ValueError("Number of columns do not match expected size of kernel matrix")

        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    continue
                
                val = self.fnc(x[:,i], x[:,j], self.theta)
                self.mat[i,j] = val
                self.mat[j,i] = val

    def compute_asym(self, x1:NDArray[np.float64], x2:NDArray[np.float64]):
        if (x1.shape[1] != self.n):
            raise ValueError("Number of columns do not match expected size of kernel matrix")
        
        outmat = np.zeros((self.n,x2.shape[1]))
        # print(outmat.shape)
        for i in range(self.n):
            for j in range(x2.shape[1]):
                val = self.fnc(x1[:,i], x2[:,j], self.theta)
                outmat[i,j] = val
        
        return outmat

    # compute dK / dtheta for every index (theta will be a vector of parameters and each one will have a different derivative matrix)
    def compute_dKdtheta(self, x:NDArray[np.float64]):
        # return a tensor where each 2d slice is a dK/dtheta matrix
        if x.shape[1] != self.n:
            raise ValueError("Number of columns do not match expected size of kernel matrix")

        outtens = np.zeros((self.n, self.n, len(self.theta)))
        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    continue
                
                derivs = self.fnc_p(x[:,i], x[:,j], self.theta)
                outtens[i,j,:] = derivs

        return outtens


    def __repr__(self):
        s1 = f"Kernel object with dimension {self.n}x{self.n} and matrix:\n"
        s2 = np.array2string(self.mat)
        return s1 + s2

class GPR:
    def __init__(self, n: int, d: int, kername: str, noise: float, use_ns: bool = False):
        self.n = n # number of data samples
        self.d = d # dimension of data
        self.sig = noise # estimate for noise variance
        self.use_ns = use_ns

        match kername:
            case "RBF":
                def ker(x, y, theta):
                    # theta[0] = sigma_f
                    # theta[1] = l
                    t = np.dot(x-y, x-y)
                    out = theta[0]**2 * math.exp(-1/(2*theta[1]) * t)
                    return out

                def ker_p(x, y, theta):
                    # theta[0] = sigma_f
                    # theta[1] = l
                    out = np.zeros(2)
                    t = -1/(2*theta[1]) * np.dot(x-y,x-y)
                    out[0] = 2*theta[0] * math.exp(t)
                    out[1] = np.dot(x-y, x-y) * 1 / (2*theta[1]**2) * theta[0]**2 * math.exp(t)
                    return out

                ker_obj = Kernel(ker, n, ker_p, 2)
                self.ker = ker_obj
            
            case _:
                raise TypeError("Unrecognized kernel name")

    # add the input data for the regression (does not train the hyperparameters, just adds input data for a given set of hyperparameters
    def add_inputs(self, x:NDArray[np.float64], y:NDArray[np.float64]):
        if (x.shape[0] == self.n and x.shape[1] == self.d):
            self.X = x.T
        elif (x.shape[0] == self.d and x.shape[1] == self.n):
            self.X = x
        else:
            raise ValueError(f"Input data is wrong dimension. Should be {self.d} by {self.n}")

        if (len(y) != self.n):
            raise ValueError(f"Input data is wrong dimension. y should be of length {self.n}")

        self.y = y

    def set_theta(self, theta:NDArray[np.float64]):
        # set the hperparameters for the kernel
        self.ker.setTheta(theta)

    # create the kernel matrix from the given input data
    def create_K(self):
        # assumes that the input data has already been initialized
        self.ker.compute(self.X)

    # add data and fit to the data in one function
    def fit(self, x:NDArray[np.float64], y:NDArray[np.float64], theta:NDArray[np.float64]):
        self.set_theta(theta)
        self.add_inputs(x, y)
        self.create_K()

    # predict (after K has already been made)
    def predict_aft_K(self, x:NDArray[np.float64]):
        if x.ndim == 1:
            x = np.array([x])
            # print(x.shape)
        # assumes that K and X have already been initialized
        K_star =self.ker.compute_asym(self.X, x)
        Khat = self.ker.mat + self.sig**2 * np.eye(self.n)
        if self.use_ns:
            G0 = Khat / (np.linalg.norm(Khat, 1) * np.linalg.norm(Khat, np.inf))
            Khati, _, _ = Newton_Shulz(Khat, initial_guess=G0)
        else:
            Khati = np.linalg.inv(Khat)
        return K_star.T @ Khati @ self.y


    # compute the derivative of the NEGATIVE log liklihood (for gradient descent optimization)
    def log_like_der(self):
        # formula is tr( (Khati - (Khati * y)(Khati * y)^T) * dKhat/dtheta)
        # create the derivative matrices 
        der_tens = self.ker.compute_dKdtheta(self.X) # nxnxt tensor where t is the number of parameters in theta vector
        Khat = self.ker.mat + self.sig**2 * np.eye(self.n)
        if self.use_ns:
            G0 = Khat / (np.linalg.norm(Khat, 1) * np.linalg.norm(Khat, np.inf))
            Khati, _, _ = Newton_Shulz(Khat, initial_guess=G0)
        else:
            Khati = np.linalg.inv(Khat)

        t = len(self.ker.theta)
        outvec = np.zeros(t)
        for i in range(t):
            # term1 = 1/2 * self.y.T @ Khati @ der_tens[:,:,i] @ Khati @ self.y
            # term2 = 1/2 * np.trace(Khati @ der_tens[:,:,i])
            # outvec[i] = term1 - term2
            m1 = np.outer(Khati @ self.y, Khati @ self.y)
            outvec[i] = np.trace((Khati - m1) @ der_tens[:,:,i])

        return outvec

    # compute the log liklihood for the gaussian process
    def log_like(self):
        # formula is -1/2 * (y^T * Khati * y + tr(log(Khat)) + n*log(2pi)
        Khat = self.ker.mat + self.sig**2 * np.eye(self.n)
        if self.use_ns:
            G0 = Khat / (np.linalg.norm(Khat, 1) * np.linalg.norm(Khat, np.inf)) # try using identity matrix as initial guess
            Khati, _, _ = Newton_Shulz(Khat, initial_guess=G0)
        else:
            Khati = np.linalg.inv(Khat)
        logKhat = scil.logm(Khat)
        term1 = self.y.T @ Khati @ self.y 
        term2 = np.trace(logKhat) 
        term3 = self.n*np.log(2*np.pi)
        return -1/2 * (term1 + term2 + term3)

    # do gradient descent to find the best theta in terms of the log liklihood
    def grad_dec_theta(self, theta0, alpha:float, eps:float = 1e-5, return_theta:bool = False, max_iter:int = 1000):
        print("Running gradient descent with theta0 = ", theta0)
        print(f"Using a stopping condition of ||nabla f||^2 < {eps}")
        print(f"Using a step size of {alpha}")
        theta = np.copy(theta0) # for storing iterations
        self.set_theta(theta)
        self.ker.compute(self.X)
        all_theta = []
        all_theta.append(theta)

        nabla = np.zeros(self.ker.theta.shape)
        nabla = self.log_like_der()
        iter = 0
        while (np.linalg.norm(nabla) > eps):
            print(f"Iteration {iter}")
            iter += 1
            if (iter > max_iter):
                break
            print("theta = ", theta)
            print("log liklihood = ", self.log_like())
            nabla = self.log_like_der()
            theta = theta - alpha * nabla
            self.set_theta(theta)
            self.ker.compute(self.X)

            if return_theta:
                all_theta.append(theta)

        print("Optimization complete")
        if return_theta:
            return all_theta

        
            
