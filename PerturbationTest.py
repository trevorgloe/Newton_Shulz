"""
Perturbation Testing

This module tests how much a matrix 
can be perturbed before Newton-Schulz diverges when using the original (unperturbed)
matrix's inverse as the initial guess.

Experiment explanation:
1. Generate random matrix R, compute R_inv using numpy
2. Perturb R: R_perturbed = R + x * P 
   - P is a random perturbation matrix (maybe with a set norm or else it'll essentially have it's own scaling already without x)
   - x is a scaling factor (0 to max_perturbation)
3. Run Newton-Schulz on R_perturbed using R_inv as initial guess
4. Find the threshold x where it stops converging (diverges)
5. Repeat for many random matrices to find patterns

Chart Explanations:
1. "Robustness: How Much Can Matrix Change?" (Left - Scatter Plot)
   - X-axis: Condition number of the original matrix R
   - Y-axis: Maximum perturbation value where Newton-Schulz still converges
   - What it shows: Whether matrices with higher condition numbers (more ill-conditioned)
     can handle less perturbation before diverging. If points trend downward, it means
     ill-conditioned matrices are less robust (basically that they can't handle big perturbations). Each point is one test matrix.

2. "Convergence vs Perturbation" (Right - Line Plot)
   - X-axis: Perturbation size (x value from R + x*P)
   - Y-axis: Divergence/Convergence speed (still trying to adjust)
   - What it shows: For each random perturbation matrix P with perturbation size as the x-axis, when/how fast does it converge
"""

# spectral radius of convergence,
# only meant to be a thing between 0 and 1 perturbation

import numpy as np
import matplotlib.pyplot as plt
import matrix_functions as mf


def test_perturbation_robustness(n_matrices=10, matrix_size=50, max_perturbation=1.0, n_perturbations=50):
    """
    Test how much a matrix can be perturbed before Newton-Schulz diverges.
    
    Args:
        n_matrices: Number of random matrices to test (int)
        matrix_size: Size of test matrices, creates matrix_size x matrix_size (int)
        max_perturbation: Maximum perturbation value to test (float)
        n_perturbations: Number of perturbation values to test per matrix (int)
    
    Returns:
        List of results, each is a dict with:
        - 'matrix_idx': int
        - 'condition_number': float  
        - 'threshold': float
        - 'convergence_data': list of dicts
    """
    print("="*60)
    print("TESTING PERTURBATION")
    print("="*60)
    print(f"Testing {n_matrices} random {matrix_size}x{matrix_size} matrices")
    print(f"Perturbation range: 0 to {max_perturbation}")
    print(f"Testing {n_perturbations} perturbation values per matrix")
    
    results = []
    perturbation_values = np.linspace(0, max_perturbation, n_perturbations)
    
    for matrix_idx in range(n_matrices):
        print(f"\n--- Matrix {matrix_idx + 1}/{n_matrices} ---")
        
        # Generate random matrix R with varying condition numbers
        # HOW WE PICK POINTS FOR LEFT GRAPH:
        # - We create n_matrices different random matrices (one per point on left graph)
        # - Each matrix gets a different "scaling factor" to control its condition number
        # - Formula: scaling_factor = 0.1 + (matrix_idx / n_matrices) * 2.9
        #   This gives scaling factors from 0.1 (worst conditioned) to 3.0 (best conditioned)
        # - We build: R = random_matrix + scaling_factor * identity_matrix
        # - Smaller scaling_factor → matrix closer to singular → higher condition number
        # - Larger scaling_factor → matrix more stable → lower condition number
        # - Each point on left graph = one matrix with its condition number (x) and threshold (y)
        np.random.seed(matrix_idx)
        scaling_factor = 0.1 + (matrix_idx / n_matrices) * 2.9  # Range: 0.1 to 3.0
        R = np.random.randn(matrix_size, matrix_size) + scaling_factor * np.eye(matrix_size) # can divide R by its norm so its magnitude 1, adding identity reduces perturbation
        
        # Compute inverse
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError: # don't use as much, this is avoidable, and not necessarily singular
            print(f"  Matrix {matrix_idx + 1} is singular, skipping...")
            continue
        
        cond_num = float(np.linalg.cond(R))
        print(f"  Condition number: {cond_num:.6e}")
        
        # Generate perturbation matrix
        P = np.random.randn(matrix_size, matrix_size)
        # magnitude of x * P, classical theory: if norm(x*P)<1, the you are GAURANTEED to converge; any orthogonal matrix has cond 1, not just I. maybe try to test
        # look at stats of random matrices not the individual matrix to find trend
        # many matries of same cond and random perturbation
        # uniform distribution might have bad cond
        # normally distributed, sqrt n is approx norm
        # how to pick othorgonal matrix correctly
        
        # Test each perturbation value
        convergence_data = []
        for x in perturbation_values:
            R_perturbed = R + x * P
            
            # Check if matrix is still invertible
            try:
                cond_perturbed = float(np.linalg.cond(R_perturbed))
                if np.isinf(cond_perturbed) or cond_perturbed > 1e15:
                    convergence_data.append({
                        'perturbation': float(x),
                        'converged': False,
                        'reason': 'singular',
                        'iterations': 0,
                        'final_error': float('inf'),
                        'cond_perturbed': cond_perturbed
                    })
                    continue
            except Exception:
                convergence_data.append({
                    'perturbation': float(x),
                    'converged': False,
                    'reason': 'singular',
                    'iterations': 0,
                    'final_error': float('inf'),
                    'cond_perturbed': float('inf')
                })
                continue
            
            # Run Newton-Schulz
            try:
                G, allG, iterations = mf.Newton_Shulz(
                    R_perturbed,
                    loops=2000,
                    initial_guess=R_inv.copy(),
                    convergence_threshold=1e-8
                )
                
                final_error = float(np.linalg.norm(R_perturbed @ G - np.eye(matrix_size)))
                converged = final_error < 1e-6 and iterations < 2000
                
                convergence_data.append({
                    'perturbation': float(x),
                    'converged': bool(converged),
                    'reason': 'converged' if converged else 'diverged',
                    'iterations': int(iterations + 1),
                    'final_error': final_error,
                    'cond_perturbed': cond_perturbed
                })
                
            except Exception as e:
                convergence_data.append({
                    'perturbation': float(x),
                    'converged': False,
                    'reason': str(e)[:50],
                    'iterations': 0,
                    'final_error': float('inf'),
                    'cond_perturbed': cond_perturbed
                })
        
        # Find threshold
        converged_perturbations = [d['perturbation'] for d in convergence_data if d['converged']]
        threshold = float(max(converged_perturbations)) if converged_perturbations else 0.0
        
        results.append({
            'matrix_idx': int(matrix_idx),
            'condition_number': cond_num,
            'threshold': threshold,
            'convergence_data': convergence_data
        })
        
        print(f"  Convergence threshold: {threshold:.6f}")
    
    return results


def plot_perturbation_results(perturbation_results):
    """
    Plot the results of perturbation robustness testing.
    
    Creates 4 plots showing different aspects of the results:
    1. Threshold vs condition number: Does robustness depend on condition number?
    2. Convergence pattern: How does convergence change with perturbation?
    3. Final error vs perturbation: How does error grow as perturbation increases?
    4. Threshold distribution: What's the spread of convergence thresholds?
    
    Args:
        perturbation_results: List of dicts from test_perturbation_robustness()
    
    Returns:
        None (displays plots)
    """
    if not perturbation_results:
        print("No perturbation results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Convergence threshold vs condition number (ALL matrices)
    # HOW POINTS ARE SELECTED:
    # - Each point = one test matrix we generated
    # - X-axis (condition number): Computed from the matrix using np.linalg.cond()
    #   Higher condition number = matrix is closer to singular = harder to invert
    # - Y-axis (threshold): Maximum perturbation where Newton-Schulz still converged
    #   Found by testing many perturbation values and finding where it first diverges
    # - We test n_matrices different matrices (default 100), each with different condition numbers
    # - Points are evenly distributed across condition number range via scaling factors
    thresholds = [r['threshold'] for r in perturbation_results]
    cond_nums = [r['condition_number'] for r in perturbation_results]
    
    axes[0].scatter(cond_nums, thresholds, alpha=0.6, s=30, edgecolors='black', linewidths=0.3)
    axes[0].set_xlabel('Condition Number (higher = more ill-conditioned)', fontsize=11)
    axes[0].set_ylabel('Max Perturbation Before Divergence', fontsize=11)
    axes[0].set_title('Robustness: Higher Condition Number → Less Robust', fontsize=12)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    # Add text annotation explaining the relationship
    axes[0].text(0.05, 0.95, 'Lower threshold = less robust\n(diverges with smaller changes)', 
                transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Divergence speed vs perturbation
    # Y-axis: Inverted iterations (2000 - iterations) so higher = faster divergence
    # This makes it clearer: higher on y-axis = diverged faster
    # Using log scale to better see differences in divergence speed
    axes[1].set_xlabel('Perturbation Size (x)', fontsize=11)
    axes[1].set_ylabel('Divergence Speed (higher = diverged faster, log scale)', fontsize=11)
    axes[1].set_title('How Fast Does It Diverge? (Sample Matrices)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    max_iterations = 2000  # Max iterations from Newton-Schulz
    
    for i, result in enumerate(perturbation_results[:10]):  # Show more matrices
        perturbations = [d['perturbation'] for d in result['convergence_data']]
        iterations = [d['iterations'] for d in result['convergence_data']]
        converged_flags = [d['converged'] for d in result['convergence_data']]
        
        # Calculate divergence speed: inverted iterations (higher = faster divergence)
        # For diverged: speed = max_iterations - iterations (higher = diverged sooner)
        # For converged: speed = 0 (didn't diverge) - but log scale can't handle 0, so we skip them
        diverged_x = [p for p, c in zip(perturbations, converged_flags) if not c]
        diverged_speed = [max_iterations - it for it, c in zip(iterations, converged_flags) if not c]
        converged_x = [p for p, c in zip(perturbations, converged_flags) if c]
        
        # Plot diverged points in red (main focus - shows divergence speed)
        # Only plot diverged cases since log scale can't handle 0 (converged cases)
        if diverged_x and diverged_speed:
            # Filter out any zero or negative values for log scale
            valid_x = [x for x, s in zip(diverged_x, diverged_speed) if s > 0]
            valid_speed = [s for s in diverged_speed if s > 0]
            if valid_x:
                axes[1].plot(valid_x, valid_speed, 'o-', color='red', 
                            label=f"Matrix {i+1} (diverged)", alpha=0.7, markersize=4, linewidth=1.5)
        
        # Note: Converged cases (speed=0) are not plotted on log scale
        # They would appear at negative infinity, so we skip them
    
    # Only show legend for first few to avoid clutter
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:6], labels[:6], loc='best')
    
    # Plot 3: Final error vs perturbation (for converged cases only) - COMMENTED OUT
    # axes[1, 0].set_xlabel('Perturbation Size (x)', fontsize=11)
    # axes[1, 0].set_ylabel('Final Error (log scale)', fontsize=11)
    # axes[1, 0].set_title('Final Error vs Perturbation', fontsize=12)
    # axes[1, 0].set_yscale('log')
    # axes[1, 0].grid(True, alpha=0.3)
    # 
    # for i, result in enumerate(perturbation_results[:3]):
    #     perturbations = [d['perturbation'] for d in result['convergence_data'] if d['converged']]
    #     errors = [d['final_error'] for d in result['convergence_data'] if d['converged']]
    #     if perturbations:
    #         axes[1, 0].plot(perturbations, errors, marker='o', label=f"Matrix {i+1}", alpha=0.7, markersize=4)
    # 
    # axes[1, 0].legend()
    
    # Plot 4: Histogram of thresholds - COMMENTED OUT
    # axes[1, 1].hist(thresholds, bins=20, edgecolor='black', alpha=0.7)
    # axes[1, 1].set_xlabel('Convergence Threshold', fontsize=11)
    # axes[1, 1].set_ylabel('Number of Matrices', fontsize=11)
    # axes[1, 1].set_title('Distribution of Convergence Thresholds', fontsize=12)
    # axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    thresholds_float = [float(t) for t in thresholds]
    print("\n" + "="*60)
    print("PERTURBATION ROBUSTNESS SUMMARY")
    print("="*60)
    print(f"Number of matrices tested: {int(len(perturbation_results))}")
    print(f"Average convergence threshold: {float(np.mean(thresholds_float)):.6f}")
    print(f"Median convergence threshold: {float(np.median(thresholds_float)):.6f}")
    print(f"Min threshold: {float(np.min(thresholds_float)):.6f}")
    print(f"Max threshold: {float(np.max(thresholds_float)):.6f}")
    print(f"Std deviation: {float(np.std(thresholds_float)):.6f}")


if __name__ == "__main__":
    """
    Example usage of perturbation robustness testing.
    
    This tests: How much can a matrix change before its inverse
    becomes a bad initial guess for Newton-Schulz?
    
    Parameters you can adjust:
    - n_matrices: How many random matrices to test (more = better stats, slower)
    - matrix_size: Size of test matrices (should match your B matrices)
    - max_perturbation: Maximum perturbation to test (start with 1.0)
    - n_perturbations: Number of perturbation values to test (more = finer resolution)
    
    The plots will show:
    1. Whether robustness depends on condition number
    2. How convergence changes with perturbation size
    3. How error grows as perturbation increases
    4. Distribution of convergence thresholds
    """
    
    # Run the perturbation robustness test
    perturbation_results = test_perturbation_robustness(
        n_matrices=100,       # Test 100 random matrices (more points for clearer pattern)
        matrix_size=50,       # 50x50 matrices (adjust to match your B matrices)
        max_perturbation=1.0, # Test perturbations from 0 to 1.0
        n_perturbations=100   # Test 100 different perturbation values (more tests per matrix)
    )
    
    # Plot the results
    plot_perturbation_results(perturbation_results)

