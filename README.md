
## Code Usage
### Hutchinson Stochastic Trace Estimator
In the [functions/stoch_trace_est.py](functions/stoch_trace_est.py) there are 2 basic Hutchinson trace estimators implemented. The first estimates 
$$
\text{tr}(A) \approx = \frac{1}{m}\sum_{j=1}^m v_j^T A v_j
$$
where $v_j$ are normalized Rademacher random vectors. This function can be called via 
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import stoch_trace_est as ste

trace_est = ste.STE(A, l=20) # assuming A has already been defined as a matrix
```
The `STE(A, l)` function assumes that `A` is square. 

The second estimator estimates the trace of `A^{-1}` via 
$$
\text{tr}(A^{-1}) \approx = \frac{1}{m}\sum_{j=1}^m v_j^T A^{-1} v_j
$$
where the $v_j$ are also normalized Rademacher random vectors. The $A^{-1}v_j$ product is computed by solving the linear system $Ax=v_j$ for $x$, using the conjugate gradient method. The function utilizes `scipy.sparse.linalg.cg` with a max iterations passed for these linear solves. This function can be called via
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import stoch_trace_est as ste

trace_est = ste.STE_kry(A, l=20, maxiter=10) # assuming A has already been defined as a matrix
```
The `STE(A, l, maxiter)` function assumes that `A` is symmetric (required for conjugate gradient algorithm). The `maxiter` parameter caps the total number of iterations that *every* conjugate gradient call uses. 
