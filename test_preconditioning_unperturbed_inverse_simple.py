import numpy as np
import matplotlib.pyplot as plt

import ilu

# Settings
N = 80
N_TRIALS = 10
MAX_ITERS = 80
PERTURB_SCALE = 1e-2
ILU_DROP_TOL = 1e-3
ILU_DIAG_EPS = 1e-12
TOL = 1e-10


def generate_invertible_matrix(n: int) -> np.ndarray:
    """Generate a random invertible matrix using randn."""
    while True:
        A = np.random.randn(n, n)
        try:
            np.linalg.inv(A)
            return A
        except np.linalg.LinAlgError:
            pass


def ns_errors_on_original_system(
    A_iter: np.ndarray,
    G0_iter: np.ndarray,
    A_original: np.ndarray,
    recover_fn=None,
    max_iters: int = MAX_ITERS,
    tol: float = TOL,
) -> np.ndarray:
    """
    Run Newton-Schulz on A_iter starting from G0_iter, but always report
    error on the original matrix A_original: ||A_original @ G - I||.

    If recover_fn is provided, it maps the iterate-space inverse approximation
    back to an approximation for A_original^{-1} before error is measured.
    """
    n = A_iter.shape[0]
    I = np.eye(n)
    G = G0_iter.copy()
    errors = []

    for _ in range(max_iters):
        G_eval = recover_fn(G) if recover_fn is not None else G
        err = np.linalg.norm(A_original @ G_eval - I, ord='fro')
        errors.append(err)

        if err < tol:
            break

        G = G @ (2 * I - A_iter @ G)

    return np.array(errors, dtype=float)


def pad_errors(err: np.ndarray, length: int) -> np.ndarray:
    if len(err) >= length:
        return err[:length]
    out = np.empty(length, dtype=float)
    out[: len(err)] = err
    out[len(err):] = err[-1]
    return out


def run_trial() -> tuple[np.ndarray, np.ndarray]:
    """
    Build an unperturbed matrix A0, perturb it to get A, then compare:
    1. No preconditioning, initial guess inv(A0)
    2. ILU preconditioning on A, with initial guess inv(preconditioned A0)

    The perturbation is intentionally kept.
    """
    A0 = generate_invertible_matrix(N)
    A = A0 + PERTURB_SCALE * np.random.randn(N, N)

    # Regenerate if the perturbed matrix becomes singular
    while True:
        try:
            np.linalg.inv(A)
            break
        except np.linalg.LinAlgError:
            A0 = generate_invertible_matrix(N)
            A = A0 + PERTURB_SCALE * np.random.randn(N, N)

    # Baseline: unperturbed inverse as initial guess
    G0_none = np.linalg.inv(A0)
    err_none = ns_errors_on_original_system(
        A_iter=A,
        G0_iter=G0_none,
        A_original=A,
    )

    # ILU preconditioning using the API from the uploaded ilu.py
    L, U = ilu.incomplete_lu(
        A,
        drop_tol=ILU_DROP_TOL,
        diag_eps=ILU_DIAG_EPS,
    )
    A_pre = ilu.precondition_matrix(A, L, U)

    # Apply the exact same preconditioning operations to A0 before inversion
    A0_pre = ilu.precondition_matrix(A0, L, U)
    G0_ilu_pre = np.linalg.inv(A0_pre)

    err_ilu = ns_errors_on_original_system(
        A_iter=A_pre,
        G0_iter=G0_ilu_pre,
        A_original=A,
        recover_fn=lambda G_pre: ilu.recover_inverse(G_pre, L, U),
    )

    return err_none, err_ilu


def main() -> None:
    all_none = []
    all_ilu = []

    for _ in range(N_TRIALS):
        e_none, e_ilu = run_trial()
        all_none.append(pad_errors(e_none, MAX_ITERS))
        all_ilu.append(pad_errors(e_ilu, MAX_ITERS))

    median_none = np.median(np.vstack(all_none), axis=0)
    median_ilu = np.median(np.vstack(all_ilu), axis=0)

    plt.figure(figsize=(8, 5))
    plt.semilogy(median_none, label='No preconditioning')
    plt.semilogy(median_ilu, label='ILU preconditioning')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|A G - I\|_F$')
    plt.title('Newton-Schulz with Unperturbed-Inverse Initial Guess')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
