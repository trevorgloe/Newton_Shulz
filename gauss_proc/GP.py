import numpy as np
from typing import Callable
from numpy.typing import NDArray
import math

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


    def __repr__(self):
        s1 = f"Kernel object with dimension {self.n}x{self.n} and matrix:\n"
        s2 = np.array2string(self.mat)
        return s1 + s2

class GPR:
    def __init__(self, n: int, d: int, kername: str, noise: float):
        self.n = n # number of data samples
        self.d = d # dimension of data
        self.sig = noise # estimate for noise variance

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
                    out = np.zeros((1,2))
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

    # predict (after K has already been made)
    def predict_aft_K(self, x:NDArray[np.float64]):
        if x.ndim == 1:
            x = np.array([x])
            # print(x.shape)
        # assumes that K and X have already been initialized
        K_star =self.ker.compute_asym(self.X, x)
        Khat = self.ker.mat + self.sig**2 * np.eye(self.n)
        return K_star.T @ np.linalg.inv(Khat) @ self.y


