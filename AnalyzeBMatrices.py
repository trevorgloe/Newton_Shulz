import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import matrix_functions as mf

def load_B_matrices(data_folder="B_Binv_Mike"):
    data_path = Path(data_folder)
    
    B_matrices = {}
    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.npy':
            filename = file_path.stem # gets rid of .npy
            if filename.startswith('B_') and not filename.startswith('B_inv'):
                parts = filename.split('_')
                if len(parts) == 2:
                    matrix_num = parts[1]
                    matrix = np.load(file_path)
                    matrix = matrix.squeeze()
                    B_matrices[matrix_num] = matrix
    
    # Load B_inv matrices
    B_inv_matrices = {}
    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.npy':
            filename = file_path.stem
            if filename.startswith('B_inv_'):
                parts = filename.split('_')
                if len(parts) == 3:
                    matrix_num = parts[2]
                    matrix = np.load(file_path)
                    matrix = matrix.squeeze()
                    B_inv_matrices[matrix_num] = matrix
    
    return B_matrices, B_inv_matrices


# Running Newton Schulz & Compting Errors

def compute_errors_per_iteration(A, allG):
    I = np.eye(A.shape[0])
    errors = []
    
    for G in allG:
        AG = A @ G
        
        error = np.linalg.norm(AG - I)
        errors.append(error)
    
    return errors


def run_newton_schulz_with_errors(A, matrix_name=""):
    print(f"Running Newton-Schulz on {matrix_name}")
    
    G, allG, iterations = mf.Newton_Shulz(A)
    
    errors = compute_errors_per_iteration(A, allG)
    
    # Print the final error with 6 digits after decimal
    final_error = errors[-1]
    print(f"Final error: {final_error:.6e}")
    print(f"Total iterations: {iterations + 1}")
    
    # Also validate using the existing validate function
    error_msg, error_val = mf.validate(A, G)
    print(f"Validation: {error_msg}{error_val:.6e}")
    
    return G, allG, errors


def analyze_matrix_relationships(matrices):
    print("Matrix relations")
    
    matrix_names = sorted(matrices.keys(), key=lambda x: int(x.split('_')[-1]))
    
    # Dictionary to store relationship data
    relationships = {}
    
    # Compare each matrix with the next one
    for i in range(len(matrix_names) - 1):
        name1 = matrix_names[i]
        name2 = matrix_names[i + 1]
        
        A1 = matrices[name1]
        A2 = matrices[name2]
        
        # Compute the difference matrix
        # If the matrices are related, the difference might be small or have some pattern
        diff = A2 - A1
        
        # Compute the norm of the difference (overall magnitude of change)
        diff_norm = np.linalg.norm(diff)
        
        # Also compute relative difference (normalized by size of matrices)
        # This tells us if the change is big or small relative to the matrix size
        A1_norm = np.linalg.norm(A1)
        relative_diff = diff_norm / A1_norm if A1_norm > 0 else 0
        
        # Store the relationship data
        relationships[f"{name1} -> {name2}"] = {
            "diff_norm": diff_norm,
            "relative_diff": relative_diff,
            "diff_matrix": diff
        }
        
        # Print the relationship
        print(f"\n{name1} -> {name2}:")
        print(f"  Absolute difference norm: {diff_norm:.6e}")
        print(f"  Relative difference: {relative_diff:.6e}")
        
        # Check if the difference is very small (might indicate convergence)
        if relative_diff < 1e-10:
            print(f"  Note: Matrices are very similar! (relative diff < 1e-10)")
        elif relative_diff < 1e-6:
            print(f"  Note: Matrices are quite similar (relative diff < 1e-6)")
    
    return relationships


def compute_matrix_properties(matrices):
    
    print("Matrix Properties")
    
    properties = {}
    
    for name, A in matrices.items():
        print(f"\n{name}:")
        
        # Basic properties
        print(f"  Shape: {A.shape}")
        
        # Condition number: tells us how "close to singular" the matrix is
        # Ratio of how big A can scale smth to how small it can scale smth
        # Large condition number = matrix is close to singular = hard to invert accurately
        try:
            cond_num = np.linalg.cond(A)
            # Ensure cond_num is a scalar (convert to float if it's an array)
            cond_num = float(cond_num) if isinstance(cond_num, np.ndarray) else cond_num
            
            if np.isinf(cond_num) or np.isnan(cond_num):
                print(f"  Condition number: INFINITE (matrix is singular/not invertible)")
                print(f"    This means the matrix cannot be inverted. Check if the matrix is actually singular.")
            else:
                print(f"  Condition number: {cond_num:.6e}")
                if cond_num > 1e12:
                    print(f"    Warning: Very large condition number! Matrix is nearly singular.")
                elif cond_num > 1e8:
                    print(f"    Note: Large condition number. Matrix is ill-conditioned.")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  Condition number: ERROR - {e}")
            cond_num = np.inf
        
        # Store properties
        properties[name] = {
            "shape": A.shape,
            "condition_number": cond_num
        }
    
    return properties

def plot_convergence(errors_dict, title="Newton-Schulz Convergence"):
    plt.figure(figsize=(10, 6))
    
    # Plot error for each matrix
    for matrix_name, errors in errors_dict.items():
        # errors is a list: [error_iter_0, error_iter_1, error_iter_2, ...]
        iterations = range(len(errors))
        plt.plot(iterations, errors, marker='o', label=matrix_name, linewidth=2, markersize=4)
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Error ||A @ G - I||", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale because errors can vary a lot
    plt.tight_layout()
    plt.show() #interrupts program until you close it
    
    print(f"\nPlotted convergence for {len(errors_dict)} matrices")


def plot_matrix_relationships(relationships):
    # look at random matrices
    # look at their differences and the limit of when the matrix causes it diverge
    # use numpy inverse to get Random1, R_inv_1
    # 'perturb' matrix with a small multiple (x) of one matrix. Use Newton-Shulz until it diverges (Original guess is R_inv_1)
    #  Try to find when x until it converges
    # many original R's and many perturbation values and see when it diverges
    # chance or cutoff?
    # Condition number: tells us how "close to singular" the matrix is
        # Ratio of how big A can scale smth to how small it can scale smth
        # Large condition number = matrix is close to singular = hard to invert accurately

    """
    Plot how matrices change from one to the next.
    
    This shows the relationship between successive matrices.
    """
    if not relationships:
        print("No relationships to plot")
        return
    
    names = list(relationships.keys())
    diff_norms = [relationships[name]["diff_norm"] for name in names]
    
    plt.figure(figsize=(12, 6))
    plt.plot(diff_norms, marker='o', linewidth=2, markersize=6)
    plt.xlabel("Matrix Pair (consecutive)", fontsize=12)
    plt.ylabel("Difference Norm ||B_{i+1} - B_i||", fontsize=12)
    plt.title("How Much Each Matrix Differs from the Previous One", fontsize=14)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPlotted relationships for {len(relationships)} matrix pairs")


def main():

    print("="*60)
    print("ANALYZING B MATRICES FROM FOLDER")
    print("="*60)

    print("\n[Step 1] Loading B and B_inv matrices from B_Binv_Mike folder...")
    print("  - Loading B_*.npy files (matrices to invert)")
    print("  - Loading B_inv_*.npy files (known inverses, used as initial guesses)")
    print("  - Squeezing 3D arrays to 2D (removing extra dimensions)")
    B_matrices, B_inv_matrices = load_B_matrices("B_Binv_Mike")
    
    if not B_matrices:
        print("ERROR: No B matrices found! Check that the B_Binv_Mike folder exists and has .npy files.")
        return
    
    print("\n[Step 2] Computing matrix properties (condition numbers) for B matrices...")
    print("  - Condition number measures how sensitive matrix is to numerical errors")
    print("  - Infinite condition number means matrix is singular (cannot be inverted)")
    # Convert B_matrices dict to format expected by compute_matrix_properties
    B_matrices_for_props = {f"B_{num}": B for num, B in B_matrices.items()}
    properties = compute_matrix_properties(B_matrices_for_props)
    
    print("\n[Step 3] Analyzing relationships between consecutive B matrices...")
    print("  - Comparing each B matrix with the next one (B_1 vs B_2, etc.)")
    print("  - Computing how much each matrix differs from the previous one")
    relationships = analyze_matrix_relationships(B_matrices_for_props)
    
    # For B_1: use diagonal initialization
    # For B_2 through B_10: use B_inv_{x-1} as initial guess
    print("\n[Step 4] Running Newton-Schulz on B matrices")
    print("  - B_1: Using diagonal initialization (original method)")
    print("  - B_2 through B_10: Using B_inv_{x-1} as initial guess")
    print("  - Tracking convergence: stops when ||B @ G - I|| < 1e-8 or max iterations reached")
    results = {}
    errors_dict = {}
    
    test_numbers = sorted([num for num in B_matrices.keys() if num.isdigit()], 
                        key=lambda x: int(x))[:10]  # First 10 matrices
    
    print(f"\nTesting on {len(test_numbers)} B matrices: B_{', '.join(test_numbers)}")
    
    for num in test_numbers:
        B_name = f"B_{num}"
        B = B_matrices[num]
        
        # Determine initial guess
        if num == "1":
            # For B_1, use diagonal initialization (original method)
            initial_guess = None
            print(f"\n{B_name}: Using diagonal initialization")
        else:
            # For B_x (x > 1), use B_inv_{x-1} as initial guess
            prev_num = str(int(num) - 1)
            if prev_num in B_inv_matrices:
                initial_guess = B_inv_matrices[prev_num]
                print(f"\n{B_name}: Using B_inv_{prev_num} as initial guess")
            else:
                print(f"\n{B_name}: Warning - B_inv_{prev_num} not found, using diagonal initialization")
                initial_guess = None
        
        # Run Newton-Schulz with initial guess
        G, allG, iterations = mf.Newton_Shulz(B, loops=2000, initial_guess=initial_guess)
        
        # Compute errors for each iteration
        errors = compute_errors_per_iteration(B, allG)
        
        # Validate final result
        error_msg, final_error = mf.validate(B, G)
        
        print(f"  Final {error_msg}{final_error:.6e}")
        print(f"  Total iterations: {iterations + 1}")
        
        # Store results
        results[B_name] = {
            "G": G,
            "allG": allG,
            "errors": errors,
            "iterations": iterations + 1,
            "used_initial_guess": initial_guess is not None
        }
        errors_dict[B_name] = errors
    
    print("\n[Step 5] Plotting results...")
    plot_convergence(errors_dict, title="Newton-Schulz Convergence on B Matrices")
    plot_matrix_relationships(relationships)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return results, relationships, properties

if __name__ == "__main__":
    # Run the main analysis
    result = main()
    if result is not None:
        results, relationships, properties = result
    
    # You can also test on individual matrices here if you want
    # For example:
    # matrices = load_B_matrices("B_Binv_Mike")
    # B1 = matrices["B_inv_1"]
    # G1, allG1, errors1 = run_newton_schulz_with_errors(B1, "B_inv_1")
    # plot_convergence({"B_inv_1": errors1})


