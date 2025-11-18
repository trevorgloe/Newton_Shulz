import os
import numpy as np
import matplotlib.pyplot as plt

import matrix_functions as mf

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

def generate_random_matrix(size: int = 2048, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((size, size))
    return data @ data.T + size * np.eye(size)


def run_newton_shulz_case(
    matrix: np.ndarray,
    label: str,
    *,
    loops: int = 20,
    collect_errors: bool = True,
):
    print("condition number: ",np.linalg.cond(matrix))
    print(f"\nTesting Newton-Schulz with {label}:")
    approx_inv, history, iterations = mf.Newton_Shulz(matrix, loops=loops)
    print(mf.validate(matrix, approx_inv))
    if collect_errors:
        errors = mf.compute_errors_per_iteration(matrix, history)
        print(f"{label} errors: {errors}")
        print("Interations: ", iterations)
        return errors
    return None


def main():
    errors = {}

    result = run_newton_shulz_case(A1, "A1 (3x3 matrix)")
    if result is not None:
        errors["A1 (3x3)"] = result

    result = run_newton_shulz_case(A4, "A4 (13x13 matrix)")
    if result is not None:
        errors["A4 (13x13)"] = result

    data_folder = "mike_data"
    for file_name in ("B_inv_1.npy", "B_inv_2.npy"):
        matrix = np.load(os.path.join(data_folder, file_name)).squeeze()
        label = f"{file_name} (loaded)"
        result = run_newton_shulz_case(matrix, label)
        if result is not None:
            errors[label] = result

    random_matrix = generate_random_matrix()
    run_newton_shulz_case(
        random_matrix,
        "Random 2048x2048 matrix",
        loops=10,
        collect_errors=False,
    )

    plt.figure(figsize=(10, 6))
    for label, err_values in errors.items():
        if err_values is None:
            continue
        plt.plot(err_values, marker="o", linewidth=2, markersize=4, label=label)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Error ||A @ G - I||", fontsize=12)
    plt.title("Newton-Schulz Convergence: Error vs Iteration", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
