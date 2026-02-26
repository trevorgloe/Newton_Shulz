import time
from typing import List, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray


def split(A: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    D = np.diag(np.diag(A))
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)
    return (D, L, U)


def Gauss_Seidel(A: ArrayLike, b: ArrayLike, loops: int = 100) -> Tuple[ArrayLike, List[ArrayLike]]:
    start_time = time.perf_counter()
    (D, L, U) = split(A)
    M = D - L
    N = U

    allx: List[ArrayLike] = []
    x = np.zeros_like(b)
    allx.append(x)

    for _ in range(loops):
        x = np.linalg.solve(M, N @ x + b)
        allx.append(x)

    elapsed_time = time.perf_counter() - start_time
    print(f"Gauss_Seidel took {elapsed_time:.4f} seconds to execute.")
    return (x, allx)


def direct(A: ArrayLike, b: ArrayLike) -> ArrayLike:
    start_time = time.perf_counter()
    x = np.linalg.solve(A, b)
    elapsed_time = time.perf_counter() - start_time
    print(f"Direct took {elapsed_time:.4f} seconds to execute.")
    return x


def Newton_Shulz(A: ArrayLike, loops: int = 2000, initial_guess: ArrayLike | None = None, convergence_threshold: float = 1e-8, verbose: bool = True) -> Tuple[ArrayLike, List[ArrayLike], int]:
    start_time = time.perf_counter()
    I = np.eye(A.shape[0])

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # wierd logic
    if initial_guess is not None:
        # Use provided initial guess (e.g., previous inverse from B_inv_{x-1})
        G = initial_guess.copy()
        _log("Using provided initial guess for Newton-Schulz")
    else:
        # Original method: start with diagonal matrix G = diag(1/diag(A))
        # This is a simple initialization that when A is diagonally dominant
        d = np.diag(A)
        G = np.diag(np.power(d, -1))
        _log("Using diagonal initialization (original method)")
    
    allG: List[ArrayLike] = [G.copy()]
    # print("Size of A: ", A.shape)
    # print("Size of G: ", G.shape)

    for i in range(loops):
        try:
            # Newton-Schulz iteration: G_{k+1} = G_k + (I - G_k @ A) @ G_k
            G_new = G + (I - G @ A) @ G
            if np.any(np.isnan(G_new)) or np.any(np.isinf(G_new)) or np.any(G_new > 1e20):
                _log(f"Newton-Schulz diverged at iteration {i}")
                break
            G = G_new
            allG.append(G.copy())
            # Check convergence: stop when ||A @ G - I|| < convergence_threshold
            current_error = np.linalg.norm(A @ G - I)
            if current_error < convergence_threshold:
                _log(f"Newton-Schulz converged at iteration {i} (error: {current_error:.6e} < {convergence_threshold:.2e})")
                break
        except (OverflowError, RuntimeWarning):
            _log(f"Newton-Schulz overflowed at iteration {i}")
            break

    elapsed_time = time.perf_counter() - start_time
    _log(f"Newton_Shulz took {elapsed_time:.4f} seconds to execute.")
    return (G, allG, i)


def validate(A: ArrayLike, G: ArrayLike) -> Tuple[str, float]:
    value = A @ G
    I = np.eye(A.shape[0])
    p_error = np.linalg.norm(value - I)
    return ("error: ", p_error)


def compute_errors_per_iteration(A: ArrayLike, allG: Sequence[ArrayLike]) -> List[float]:
    I = np.eye(A.shape[0])
    errors: List[float] = []

    for G in allG:
        AG = A @ G
        error = np.linalg.norm(AG - I)
        errors.append(error)

    return errors


__all__ = [
    "split",
    "Gauss_Seidel",
    "direct",
    "Newton_Shulz",
    "validate",
    "compute_errors_per_iteration",
]

