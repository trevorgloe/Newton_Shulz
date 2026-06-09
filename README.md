
## Code Usage
Generate random orthogonal matrix
Creating a random orthogonal matrix, distributed according to the Haar measure (the cannonical distribution for random orthogonal matrices) can be done using the function defined in Haar.py. To call the function, use the syntax

G = Haar.randOrth(n)
where n is the dimension of the orthogonal matrix to be obtained. This will return an n by n numpy array.
### Hutchinson Stochastic Trace Estimator
In the [functions/stoch_trace_est.py](functions/stoch_trace_est.py) there are 2 basic Hutchinson trace estimators implemented. The first estimates 
```math
\text{tr}(A) \approx = \frac{1}{m}\sum_{j=1}^m v_j^T A v_j
```
where $v_j$ are normalized Rademacher random vectors. This function can be called via 
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import stoch_trace_est as ste

trace_est = ste.STE(A, l=20) # assuming A has already been defined as a matrix
```
The `STE(A, l)` function assumes that `A` is square. 

The second estimator estimates the trace of $A^{-1}$ via 
```math
\text{tr}(A^{-1}) \approx = \frac{1}{m}\sum_{j=1}^m v_j^T A^{-1} v_j
```
where the $v_j$ are also normalized Rademacher random vectors. The $A^{-1}v_j$ product is computed by solving the linear system $Ax=v_j$ for $x$, using the conjugate gradient method. The function utilizes `scipy.sparse.linalg.cg` with a max iterations passed for these linear solves. This function can be called via
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import stoch_trace_est as ste

trace_est = ste.STE_kry(A, l=20, maxiter=10) # assuming A has already been defined as a matrix
```
The `STE(A, l, maxiter)` function assumes that `A` is symmetric (required for conjugate gradient algorithm). The `maxiter` parameter caps the total number of iterations that *every* conjugate gradient call uses. 

### Conjugate Gradient Method
In [functions/cg.py](functions/cg.py) the basic conjugate gradient method is implemented. It is a method for solving 
```math
Ax = b
```
where $A\in\mathbb{R}^{n\times n}$ is a symmetric, positive-definite matrix, $x\in\mathbb{R}^n$ and $b\in\mathbb{R}^n$ are vectors. This is the very basic method, which does not include any pre-conditioning or restarting. It can be called like so 
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import cg

x = cg.ConjugateGradient(A, b, maxk=20) # assuming A and b have already been defined as numpy arrays
```
`A` and `b` must be numpy arrays. The function takes in several optional arguments: `maxk` - the maximum number of iterations, `tol` - the error tolerance before stopping, `verbose` - to toggle printing on every iteration, and `return_all_res` - to return an array of all the residuals. If `return_all_res` is toggled, then the function is called like `x, all_res = cg.ConjugateGradient(A, b, maxk=20, return_all_res=True)`. 
