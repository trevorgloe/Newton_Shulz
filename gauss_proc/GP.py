import numpy as np
from typing import Callable
from numpy.typing import NDArray

class Kernel:
    def __init__(self, fnc: Callable[[NDArray[np.float64], NDArray[np.float64]], float], n: int):
        self.fnc = fnc
        self.n = n
        self.mat = np.zeros((n,n))

    # compute the kernel matrix for a given set of n points
    def compute(self, x:NDArray[np.float64]):
        if x.shape[1] != self.n:
            raise ValueError("Number of columns do not match expected size of kernel matrix")

        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    continue

                self.mat[i,j] = self.fnc(x[:,i], x[:,j])

    def __repr__(self):
        s1 = f"Kernel object with dimension {self.n}x{self.n} and matrix:\n"
        s2 = np.array2string(self.mat)
        return s1 + s2

