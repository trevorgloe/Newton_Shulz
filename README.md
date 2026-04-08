# Newton_Schulz

## Description of Method
The first objective of this study was to evaluate the extent of effectiveness of the Newton-Schulz method when using the previous inverse as an intial guess. Essentially, we took a matrix and its inverse and perturbed it to determine whether Newton-Schulz still converges when activated with its inverse matrix. More specifically, we were trying to determine how well of an initial guess matrix A inverse was for the inverse of the peturbed matrix A. If it was a good guess, it would be enough for convergence. It is very important that a matrix converges, as the whole goal is to compute the inverse of matrix A. 

The second objective of the study was to analyze how the inverse changed when we slightly perturbed the matrix. Given Matrix A, perturbing it by changing it to A+P where P is a small perturbation can potentially make the matrix more invertible and cause for faster convergence. We are currently testing/proving the effectiveness of this modificiation. 

## Code Usage 
### Generate random orthogonal matrix
Creating a random orthogonal matrix, distributed according to the Haar measure (the cannonical distribution for random orthogonal matrices) can be done using the function defined in [Haar.py](Haar.py). To call the function, use the syntax 
```
G = Haar.randOrth(n)
```
where `n` is the dimension of the orthogonal matrix to be obtained. This will return an n by n numpy array.
