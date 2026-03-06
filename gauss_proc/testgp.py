import GP
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def ker(x, y, theta:NDArray[np.float64]):
    return np.dot(x,y)

def ker_p(x, y, theta:NDArray[np.float64]):
    return np.zeros((0,1))


n = 13
K = GP.Kernel(ker, n, ker_p, 0)

x = np.random.randn(100, n)
K.compute(x)
print(K)

sig = 0.01
x = np.random.rand(n)
# print(x)
y = np.sin(x* 2*np.pi) + sig*np.random.randn(n)
print(x.shape)
print(y.shape)

xth = np.linspace(0,1, 100)
yth = np.sin(xth * 2*np.pi)


gauss = GP.GPR(n, 1, "RBF", sig)
# gauss.set_theta(np.array([sig*10, 1e-4]))
# gauss.add_inputs(np.array([x]), y)
# gauss.create_K()
gauss.fit(np.array([x]), y, np.array([sig*10, 1e-2]))
# new_x = np.random.rand(5)
new_y = gauss.predict_aft_K(xth)

# fig = plt.figure()
# plt.scatter(x, y, label="true data")
# plt.plot(xth, yth, label="function")
# plt.plot(xth, new_y, label="predicted")
#
# plt.legend()
# plt.show()

# try optimizing
gauss.grad_dec_theta(np.array([1, 1]), alpha=3e-5)

