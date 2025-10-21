import numpy as np
import time
import matplotlib.pyplot as plt

def split(A):
    D = np.diag(np.diag(A))
    L = -np.tril(A, k = -1)
    U = -np.triu(A, k = 1)
    return (D, L, U)
    # print("D=\n", D, "\nL=\n", L, "\nU=\n", U)

def Gauss_Seidel(A, b):
    start_time = time.perf_counter()
    (D, L, U) = split(A)
    M = D - L
    N = U
    
    allx = []
    x = np.zeros_like(b) #is this right?
    allx.append(x)
    
    loops = 100
    for i in range(loops):
        x = np.linalg.solve(M, N @ x + b)
        allx.append(x)
        # Mx = Nx + b
        # M inverse is too general of a solution, can solve for any b
        #   1. Compute the right-hand side = Nx + b
        #   2. Solve the lower-triangular system M * x_new = (N @ x + b)
        # more efficient than actually solving M inv. instead we just kind of avid inverse by putting it on left side
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Gauss_Seidel took {elapsed_time:.4f}  seconds to execute.")

    return (x, allx)

def direct(A, b):
    start_time = time.perf_counter()
    
    x = np.zeros_like(b)
    
    x = np.linalg.solve(A, b)
    #   1. Compute the right-hand side = Nx + b
    #   2. Solve the lower-triangular system M * x_new = (N @ x + b)
    # more efficient than actually solving M inv. instead we just kind of avid inverse by putting it on left side

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Direct took {elapsed_time:.4f}  seconds to execute.")

    return x

def Newton_Shulz(A):
    start_time = time.perf_counter()
    I = np.eye(A.shape[0])  # Identity matrix with size of A
    
    # Very small initial guess, I'm not sure why we need to do this, but it works.
    alpha = 0.01
    G = alpha * I
    
    allG = []
    allG.append(G)
    
    loops = 20  # Even fewer iterations
    for i in range(loops):
        try:
            # Newton-Schulz iteration: G = G + (I - G@A)@G
            G_new = G + (I - G @ A) @ G
            
            # Check for overflow/underflow
            if np.any(np.isnan(G_new)) or np.any(np.isinf(G_new)):
                print(f"Newton-Schulz diverged at iteration {i}")
                break
                
            G = G_new
            allG.append(G)
            
            # Early stopping if we're getting reasonable results
            # if i > 5:  # After a few iterations, check convergence
            #     error = np.linalg.norm(A @ x - b)
            #     if error < 1e-6:  # Good enough
            #         print(f"Newton-Schulz converged at iteration {i}")
            #         break
            
        except (OverflowError, RuntimeWarning): #float(32 bits) vs double(64 bits)
            print(f"Newton-Schulz overflowed at iteration {i}")
            break
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Newton_Shulz took {elapsed_time:.4f}  seconds to execute.")

    return (G, allG)

def validate(A, G):
    value = A@G
    I = np.eye(A.shape[0])
    p_error = (np.linalg.norm(value-I))
    return("error: ", p_error)

A1 = np.array([[4., 1., 0.],
               [1., 3., 1.],
               [0., 1., 2.]])
b1 = np.array([1., 2., 0.])

A2 = np.array([[10., 1., 1.],
               [2., 10., 1.],
               [2., 2., 10.]])
b2 = np.array([12., 13., 14.])


A3 = np.array([[10., 1., 1., 2., 3., 4., 5., 6., 7., 8., 8., 9., 10.],
               [2., 10., 1., 0., 1., 5., 6., 7., 1., 4., 2., 5., 7.],
               [2., 10., 2., 0., 1., 5., 6., 7., 1., 4., 2., 5., 8.],
               [2., 10., 1., 9., 1., 5., 6., 7., 1., 4., 7., 5., 9.],
               [2., 12., 1., 0., 1., 5., 7., 7., 1., 4., 2., 5., 0.],
               [2., 14., 1., 0., 7., 5., 6., 7., 1., 4., 2., 5., 1.],
               [2., 16., 1., 0., 1., 5., 6., 7., 1., 4., 2., 5., 2.],
               [2., 10., 3., 0., 1., 5., 6., 7., 1., 4., 2., 5., 3.],
               [2., 10., 1., 0., 1., 7., 6., 7., 1., 4., 2., 5., 4.],
               [3., 10., 1., 0., 1., 4., 6., 7., 1., 4., 2., 5., 5.],
               [4., 10., 1., 0., 1., 8., 6., 7., 1., 4., 2., 5., 6.],
               [5., 10., 1., 0., 1., 9., 6., 7., 1., 4., 2., 5., 7.],
               [6., 2., 10., 10., 10., 10., 2., 2., 2., 2., 3., 4., 9.]])
b3 = np.array([12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 25.])

# print("eigen stuff: ", np.linalg.eig(A3)) # check for Minv*N, but not needed bc we shouldn't be getting Minv and if it's <= 1

A4 = np.array([
    [20., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 20., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 20., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 20., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 20., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 20., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 20., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 20., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 20., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 20., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 20., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 20., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 20.]
])

b4 = np.array([32., 32., 32., 32., 32., 32., 32., 32., 32., 32., 32., 32., 32.])

A5 = np.random.rand(10,10) # better randn for bell curve
b5 = np.random.rand(10)

#------------------------------

# Gx1 = Gauss_Seidel(A1, b1)
# Dx1 = direct(A1, b1)
# print("Test 1: \n", Gx1)
# print(Dx1)


# Gx2 = Gauss_Seidel(A2, b2)
# Dx2 = direct(A2, b2)
# print("Test 2: \n", Gx2)
# print(Dx2)


# Gx3 = Gauss_Seidel(A3, b3)
# Dx3 = direct(A3, b3)
# print("Test 3: \n", Gx3)
# print(Dx3)
# print("Gauss_Seidel: ", validate(A3,Gx3,b3))
# print("Direct: ", validate(A3,Dx3,b3))
# get array of errors

# Test with a simpler matrix first
print("Testing Newton-Schulz with A1 (3x3 matrix):")
Newx1, Newx1_all = Newton_Shulz(A1)
print(Newx1)
print(Newx1@A1)
print(validate(A1,Newx1))
# print("Newton_Shulz A1: ", validate(A1,Newx1,b1))

print("\nTesting Newton-Schulz with A4 (13x13 matrix):")
Newx4, Newx4_all = Newton_Shulz(A4)


plt.figure() # try making a plot of the errors for each iteration
plt.plot()#vector of errors
# use plt to graph errors



