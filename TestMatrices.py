from pathlib import Path
import time

import Matrices as M  # noqa: F401  # keep side effects for plotting/manual runs
import numpy as np

import matrix_functions as mf


def load_matrices():
    mats = []
    foldername = "mike_data"
    for i in range(1, 18):
        mats.append(np.squeeze(np.load(foldername + "/B_inv_" + str(i) + ".npy")))
    return mats


def test_newton_shulz_runtime_on_b_inv_1():
    """
    Load B_inv_1.npy, invert it using Newton-Schulz, and time the execution.
    """
    matrix_path = Path(__file__).with_name("mike_data") / "B_inv_1.npy"
    matrix = np.load(matrix_path)

    start_time = time.perf_counter()
    inverse, _ = mf.Newton_Shulz(matrix)
    elapsed = time.perf_counter() - start_time

    print(f"Newton-Schulz inversion of B_inv_1.npy took {elapsed:.4f} seconds.")

    assert inverse.shape == matrix.shape
    assert elapsed > 0


if __name__ == "__main__":
    mats = load_matrices()
    print(mats[0].shape)
