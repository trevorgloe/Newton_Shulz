"""
Lightweight incomplete LU (ILU) and partial pivoted LU helpers.

ILU: Keeps original nonzero pattern with drop tolerances.
prrLU-style routine: Uses complete pivoting on the leading `num_rows`
elimination steps, then stops early. This is a practical partial LU / rank-aware
baseline for preconditioning, but it is not a full strong rank-revealing LU
implementation.
"""
from typing import Tuple
import numpy as np

ArrayLike = np.ndarray


def incomplete_lu(
    A: ArrayLike,
    *,
    drop_tol: float = 1e-3,  # tolerance for dropping elements, based on absolute value < drop_tol
    diag_eps: float = 1e-12,  # Minimum allowed pivot magnitude (to avoid division by ~0)
    max_rows: int | None = None,  # Stop after processing this many rows (None = all rows)
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Incomplete LU factorization with optional early stopping.
    
    Parameters
    ----------
    A : ArrayLike
        Square matrix to factorize (not modified in place)
    drop_tol : float
        Tolerance for dropping small elements
    diag_eps : float
        Minimum allowed pivot magnitude to avoid division by ~0
    max_rows : int, optional
        Stop after processing this many rows. If None, process all rows.
        This parameter allows partial factorization similar to prrLU.
    
    Returns
    -------
    L, U : Tuple[ArrayLike, ArrayLike]
        L is unit-lower triangular, U is upper-triangular.
        If max_rows < n, the remaining rows of L are identity (L[i,i]=1)
        and U retains the original values in those rows.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

    n = A.shape[0]
    if max_rows is None:
        max_rows = n
    max_rows = min(max_rows, n)  # Ensure we don't exceed matrix size
    
    pattern_tol = drop_tol
    pattern = np.abs(A) > pattern_tol  # allowed nonzeros (no fill beyond this)

    L = np.eye(n, dtype=float)
    U = A.copy()

    for k in range(max_rows):
        pivot = U[k, k]
        if abs(pivot) < diag_eps:
            pivot = diag_eps if pivot >= 0 else -diag_eps
            U[k, k] = pivot

        for i in range(k + 1, n):
            if not pattern[i, k]:
                continue

            mult = U[i, k] / pivot

            if abs(mult) < drop_tol:
                L[i, k] = 0.0
                U[i, k] = 0.0
                continue

            L[i, k] = mult
            U[i, k] = 0.0  # enforce zero below diagonal in U

            for j in range(k + 1, n):
                if not pattern[i, j]:
                    continue
                updated = U[i, j] - mult * U[k, j]
                U[i, j] = 0.0 if abs(updated) < drop_tol else updated
    
    return (L, U)


def prrlu(
    A: ArrayLike,
    *,
    num_rows: int,
    diag_eps: float = 1e-12,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Partial pivoted LU factorization with complete pivoting on the first
    `num_rows` elimination steps.

    The routine builds permutation matrices P and Q such that the permuted
    matrix ``P @ A @ Q`` has strong pivots brought into the leading block.
    Standard Gaussian elimination is then applied only for the first
    ``num_rows`` columns. This is not a full strong RR-LU implementation, so
    in a writeup it is safer to describe it as a partial LU with complete
    pivoting / rank-aware truncated LU baseline.

    Parameters
    ----------
    A : ArrayLike
        Square matrix to factorize (not modified in place)
    num_rows : int
        Number of elimination steps. Must satisfy 1 <= num_rows <= n.
    diag_eps : float
        Minimum allowed pivot magnitude to avoid division by ~0.

    Returns
    -------
    L, U, P, Q : Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        Matrices satisfying approximately ``P @ A @ Q ≈ L @ U`` over the
        processed leading block. ``P`` and ``Q`` are permutation matrices.

    Notes
    -----
    - The first ``num_rows`` pivots are chosen by complete pivoting on the
      trailing submatrix, which is the main source of the "rank-revealing"
      behavior here.
    - Elimination then stops early, so this is best viewed as a practical
      partial LU preconditioner rather than a full RR-LU algorithm.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

    n = A.shape[0]
    if num_rows < 1 or num_rows > n:
        raise ValueError(f"num_rows must be between 1 and {n}, got {num_rows}")

    U = A.copy()
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    Q = np.eye(n, dtype=float)

    for k in range(num_rows):
        trailing = np.abs(U[k:, k:])
        rel_i, rel_j = np.unravel_index(np.argmax(trailing), trailing.shape)
        piv_i = k + rel_i
        piv_j = k + rel_j

        if piv_i != k:
            U[[k, piv_i], :] = U[[piv_i, k], :]
            P[[k, piv_i], :] = P[[piv_i, k], :]
            if k > 0:
                L[[k, piv_i], :k] = L[[piv_i, k], :k]

        if piv_j != k:
            U[:, [k, piv_j]] = U[:, [piv_j, k]]
            Q[:, [k, piv_j]] = Q[:, [piv_j, k]]

        pivot = U[k, k]
        if abs(pivot) < diag_eps:
            pivot = diag_eps if pivot >= 0 else -diag_eps
            U[k, k] = pivot

        for i in range(k + 1, n):
            mult = U[i, k] / pivot
            L[i, k] = mult
            U[i, k] = 0.0
            U[i, k + 1:] = U[i, k + 1:] - mult * U[k, k + 1:]

    return (L, U, P, Q)


def forward_substitution(L: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Solve L y = b for lower-triangular L with nonzero diagonal."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def back_substitution(U: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Solve U x = y for upper-triangular U with nonzero diagonal."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def forward_substitution_matrix(L: ArrayLike, B: ArrayLike) -> ArrayLike:
    """
    Solve L @ X = B for matrix B (column by column).
    Returns X = L^{-1} @ B
    """
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    
    if B.ndim == 1:
        return forward_substitution(L, B)
    
    n = B.shape[1]
    X = np.zeros_like(B, dtype=float)
    for j in range(n):
        X[:, j] = forward_substitution(L, B[:, j])
    return X


def back_substitution_matrix(U: ArrayLike, B: ArrayLike) -> ArrayLike:
    """
    Solve X @ U = B for matrix B (row by row).
    This is equivalent to computing B @ U^{-1}.
    
    We use the identity: (X @ U)^T = U^T @ X^T = B^T
    So we solve U^T @ X^T = B^T column by column.
    """
    U = np.asarray(U, dtype=float)
    B = np.asarray(B, dtype=float)
    
    if B.ndim == 1:
        # For a vector, this doesn't quite make sense, but handle it anyway
        return back_substitution(U, B)
    
    # Solve X @ U = B by computing X^T from U^T @ X^T = B^T
    # Using forward substitution on U^T (which is lower triangular)
    Ut = U.T
    Bt = B.T
    n = Bt.shape[1]
    Xt = np.zeros_like(Bt, dtype=float)
    
    for j in range(n):
        # Solve Ut @ x = bt where Ut is lower triangular
        Xt[:, j] = forward_substitution(Ut, Bt[:, j])
    
    return Xt.T


def precondition_matrix(
    A: ArrayLike,
    L: ArrayLike,
    U: ArrayLike,
    P: ArrayLike | None = None,
    Q: ArrayLike | None = None,
) -> ArrayLike:
    """
    Apply preconditioning.

    Without permutations this computes ``L^{-1} @ A @ U^{-1}``.
    With prrLU permutations it computes
    ``L^{-1} @ (P @ A @ Q) @ U^{-1}``, where ``P`` and ``Q`` are the row and
    column permutation matrices returned by ``prrlu``.
    """
    # L & U Matrices
    #
    A_eff = np.asarray(A, dtype=float)
    if P is not None:
        A_eff = P @ A_eff
    if Q is not None:
        A_eff = A_eff @ Q

    L_inv_A = forward_substitution_matrix(L, A_eff)
    A_precond = back_substitution_matrix(U, L_inv_A)
    return A_precond


def recover_inverse(
    G_precond: ArrayLike,
    L: ArrayLike,
    U: ArrayLike,
    P: ArrayLike | None = None,
    Q: ArrayLike | None = None,
) -> ArrayLike:
    """
    Recover the original inverse from the preconditioned inverse.

    Without permutations:
        A_p = L^{-1} @ A @ U^{-1}
        A^{-1} approx U^{-1} @ G_p @ L^{-1}

    With permutations from prrLU:
        A_p = L^{-1} @ (P @ A @ Q) @ U^{-1}
        A^{-1} approx Q @ U^{-1} @ G_p @ L^{-1} @ P
    """
    n = G_precond.shape[0]

    U_inv_G = np.zeros_like(G_precond, dtype=float)
    for j in range(n):
        U_inv_G[:, j] = back_substitution(U, G_precond[:, j])

    Lt = L.T
    G_core = np.zeros_like(U_inv_G, dtype=float)
    for i in range(n):
        G_core[i, :] = back_substitution(Lt, U_inv_G[i, :])

    if Q is not None:
        G_core = Q @ G_core
    if P is not None:
        G_core = G_core @ P
    return G_core


def solve_with_ilu(A: ArrayLike, b: ArrayLike, **ilu_kwargs) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Convenience helper: compute ILU factors of A and approximately solve Ax=b."""
    L, U = incomplete_lu(A, **ilu_kwargs)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return (x, L, U)


__all__ = [
    "incomplete_lu",
    "prrlu",
    "forward_substitution",
    "back_substitution",
    "forward_substitution_matrix",
    "back_substitution_matrix",
    "precondition_matrix",
    "recover_inverse",
    "solve_with_ilu",
]
