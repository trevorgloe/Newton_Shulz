import GP
import numpy as np

def ker(x, y):
    return np.dot(x,y)

K = GP.Kernel(ker, 20)

x = np.random.randn(100, 20)
K.compute(x)
print(K)
