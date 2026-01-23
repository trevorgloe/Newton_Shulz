import GP
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def ker(x, y, theta:NDArray[np.float64]):
    return np.dot(x,y)

def ker_p(x, y, theta:NDArray[np.float64]):
    return np.zeros((0,1))


K = GP.Kernel(ker, 20, ker_p, 0)

x = np.random.randn(100, 20)
K.compute(x)
print(K)

sig = 0.01
x = np.random.rand(20)
# print(x)
y = np.sin(x* 2*np.pi)

xth = np.linspace(0,1, 100)
yth = np.sin(xth * 2*np.pi)


gauss = GP.GPR(20, 1, "RBF", sig)
gauss.set_theta(np.array([1, 0.5]))
gauss.add_inputs(np.array([x]), y)
gauss.create_K()
new_x = np.random.rand(5)
new_y = gauss.predict_aft_K(new_x)

fig = plt.figure()
plt.scatter(x, y, label="true data")
plt.plot(xth, yth, label="function")
plt.scatter(new_x, new_y, label="predicted")

plt.legend()
plt.show()
