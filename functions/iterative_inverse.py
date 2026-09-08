from typing import List, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray

"""
Input:
    A: Input matrix.

Returns:
    D: Diagonal component of A.
    L: Strictly lower-triangular component of A.
    U: Strictly upper-triangular component of A.
"""
def split(A: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    return (D, L, U)

"""
Input:
    A: Coefficient matrix of the linear system Ax = b.
    b: Right-hand-side vector of the linear system.
    loops: Number of Gauss-Seidel iterations to perform.

Returns:
    A tuple (x, allx), where x is the final approximation to the solution
    and allx contains the approximation from each iteration, including the
    initial zero vector.
"""
def Gauss_Seidel(A: ArrayLike, b: ArrayLike, loops: int = 100) -> Tuple[ArrayLike, List[ArrayLike]]:
    (D, L, U) = split(A)
    M = D + L # The upper triangular matrix used in the formula for GS is lower triangular so we add the diagonal again
    N = -U

    allx: List[ArrayLike] = []
    x = np.zeros_like(b)
    allx.append(x)

    for _ in range(loops):
        rhs = N @ x + b
        x = np.linalg.solve(M, rhs) # Is a direct linalg solve too complex or should i use a manual solve?
        allx.append(x.copy())

    return (x, allx)

"""
Input:
    A: Coefficient matrix in the linear system Ax = b.
    b: Right-hand-side vector.

Returns:
    x: Exact solution to Ax = b.
"""
def direct(A: ArrayLike, b: ArrayLike) -> ArrayLike:
    x = np.linalg.solve(A, b)
    return x

"""
    Input:
        A: Square matrix whose inverse is being approximated.
        loops: Maximum number of Newton-Schulz iterations.
        initial_guess: Optional starting approximation for the inverse of A.
        convergence_threshold: Error below which the method is considered converged.
        verbose: If True, prints convergence information during iteration.

    Returns:
        G: Final approximation of the inverse of A.
        allG: List containing the inverse approximation from every iteration, including the initial guess.
        iterations: Number of Newton-Schulz iterations performed.
"""
def Newton_Shulz(
    A: ArrayLike,
    loops: int = 2000,
    initial_guess: ArrayLike | None = None,
    convergence_threshold: float = 1e-8,
    verbose: bool = True
) -> Tuple[ArrayLike, List[ArrayLike], int]:
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    I = np.eye(A.shape[0])

    # Use a supplied starting approximation if one is provided.
    if initial_guess is not None:
        G = np.asarray(initial_guess, dtype=float).copy()

        if G.shape != A.shape:
            raise ValueError("initial_guess must have the same shape as A.")

        if verbose:
            print("Using provided initial guess for Newton-Schulz")

    else:
        # If no gives approx, initialize G using the reciprocals of A's diagonal entries.
        d = np.diag(A)

        if np.any(d == 0):
            raise ValueError("Diagonal initialization cannot be used when A has a zero diagonal entry.")

        G = np.diag(1.0 / d)

        if verbose:
            print("Using diagonal initialization")

    # Store the initial approximation.
    allG: List[ArrayLike] = [G.copy()]

    current_error = np.linalg.norm(I - G @ A)

    iterations = 0

    for i in range(loops):

        # Newton-Schulz iteration:
        # G_(k+1) = G_k + (I - G_k A)G_k
        #           = 2G_k - G_k A G_k
        G_new = G + (I - G @ A) @ G

        # Stop if numerical overflow or divergence occurs.
        if not np.all(np.isfinite(G_new)) or np.any(np.abs(G_new) > 1e20):
            if verbose:
                print(f"Newton-Schulz diverged at iteration {i + 1}")
            break

        G = G_new
        allG.append(G.copy())
        iterations = i + 1

        # Check how close G is to A^(-1).
        current_error = np.linalg.norm(I - G @ A)

        if verbose:
            print(f"Current error = {current_error}")

        # Stop once the approximation is sufficiently accurate.
        if current_error < convergence_threshold:
            if verbose:
                print(
                    f"Newton-Schulz converged at iteration {iterations} "
                    f"(error: {current_error:.6e} < {convergence_threshold:.2e})"
                )
            break

    else:
        if verbose:
            print(
                f"Newton-Schulz exceeded the max number of iterations "
                f"and has error {current_error}"
            )

    return (G, allG, iterations)

"""
Input:
    A: Original matrix.
    G: Approximation of the inverse of A.

Returns:
    A tuple containing the label "error: " and the norm of the difference
    between A @ G and the identity matrix.
"""
def validate(A: ArrayLike, G: ArrayLike) -> Tuple[str, float]:
    # Multiply A by its approximate inverse.
    value = A @ G

    # Create the identity matrix with the same dimensions as A.
    I = np.eye(A.shape[0])

    # Measure how close A @ G is to the identity matrix.
    p_error = np.linalg.norm(value - I)

    return ("error: ", p_error)


"""
Input:
    A: Original matrix.
    allG: Sequence containing the inverse approximation from each iteration.

Returns:
    errors: List containing the error for each inverse approximation.
"""
def compute_errors_per_iteration(A: ArrayLike, allG: Sequence[ArrayLike]) -> List[float]:
    # Create the identity matrix with the same dimensions as A.
    I = np.eye(A.shape[0])

    errors: List[float] = []

    # Compute the error for each approximation of A^(-1).
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
