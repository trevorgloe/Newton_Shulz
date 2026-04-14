import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.preconditioners import incomplete_lu, solve_with_ilu


def demo_small_system():
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 0.0])

    print("\n=== Small 3x3 system (drop_tol=0) ===")
    L, U, res = incomplete_lu(A, drop_tol=0.0, compute_residual=True)
    x, _, _ = solve_with_ilu(A, b, drop_tol=0.0)
    print("||A - L U||_inf:", res)
    print("x:", x)
    print("Ax:", A @ x)

    print("\n=== Small 3x3 system (drop_tol=1e-2) ===")
    L, U, res = incomplete_lu(A, drop_tol=1e-2, compute_residual=True)
    x, _, _ = solve_with_ilu(A, b, drop_tol=1e-2)
    print("||A - L U||_inf:", res)
    print("x:", x)
    print("Ax:", A @ x)


def demo_mike_data():
    data_dir = Path(__file__).resolve().parents[1] / "mike_data"
    file_path = data_dir / "B_inv_1.npy"
    if not file_path.exists():
        print("\n[mike_data] Skipping: mike_data/B_inv_1.npy not found.")
        return

    A = np.load(file_path).squeeze()
    b = np.ones(A.shape[0])

    print("\n=== mike_data/B_inv_1.npy (drop_tol=1e-6) ===")
    L, U, res = incomplete_lu(A, drop_tol=1e-6, compute_residual=True)
    x, _, _ = solve_with_ilu(A, b, drop_tol=1e-6)
    print("||A - L U||_inf:", res)
    print("First 5 entries of Ax:", (A @ x)[:5])


def main():
    demo_small_system()
    demo_mike_data()


if __name__ == "__main__":
    main()
