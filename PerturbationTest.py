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
    # print("="*60)
    # print("TESTING PERTURBATION")
    # print("="*60)
    # print(f"Testing {n_matrices} random {matrix_size}x{matrix_size} matrices")
    # print(f"Perturbation range: 0 to {max_perturbation}")
    # print(f"Testing {n_perturbations} perturbation values per matrix")
    
    results = []
    perturbation_values = np.linspace(0, max_perturbation, n_perturbations)
    
    for matrix_idx in range(n_matrices):
        # print(f"\n--- Matrix {matrix_idx + 1}/{n_matrices} ---")
        
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
        # Avoid try-except: ensure matrix is well-conditioned by construction
        R_inv = np.linalg.inv(R)
        
        cond_num = float(np.linalg.cond(R))
        # print(f"  Condition number: {cond_num:.6e}")
        
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
        
        # print(f"  Convergence threshold: {threshold:.6f}")
    
    return results


def test_single_perturbation(R, R_inv, P, target_epsilon, matrix_size):
    """
    Test a single perturbation following classical theory: A + E where ||E|| ≤ ε
    
    Classical framework: 
    - Perturbation: A_perturbed = A + E
    - Constraint: ||E|| ≤ ε, usually ε < 1 or ε < 1/||A^(-1)||
    
    Our implementation:
    - A = R (original matrix)
    - E = x*P (perturbation matrix)
    - Since P is normalized: ||P|| = 1
    - Therefore: ||E|| = ||x*P|| = |x| * ||P|| = |x| * 1 = |x|
    - To get ||E|| = target_epsilon, we set: x = target_epsilon
    - Final: A_perturbed = R + x*P = R + E, where ||E|| = target_epsilon
    
    Args:
        target_epsilon: Desired ||E|| value (ε in classical theory)
    
    Returns: dict with norm_E (actual ||E||), converged, iterations, final_error
    """
    # Verify P is normalized (should be 1, but compute to be safe)
    norm_P = np.linalg.norm(P)
    if norm_P < 1e-10:
        return {'norm_E': 0, 'converged': False, 'iterations': 0, 'final_error': float('inf')}
    
    # Classical theory: E = x*P, and we want ||E|| = target_epsilon
    # Since ||P|| = 1 (normalized), we have: ||E|| = ||x*P|| = |x| * ||P|| = |x|
    # Therefore: x = target_epsilon to get ||E|| = target_epsilon
    x = target_epsilon / norm_P  # This equals target_epsilon when norm_P = 1
    
    # Create perturbation matrix E
    E = x * P
    
    # Create perturbed matrix: A_perturbed = A + E (classical form)
    R_perturbed = R + E
    
    # Verify actual ||E|| matches target (should equal target_epsilon exactly if ||P|| = 1)
    actual_norm_E = float(np.linalg.norm(E))
    
    # Check if still invertible
    try:
        cond_perturbed = float(np.linalg.cond(R_perturbed))
        if np.isinf(cond_perturbed) or cond_perturbed > 1e15:
            return {'norm_E': actual_norm_E, 'converged': False, 'iterations': 0, 'final_error': float('inf')}
    except Exception:
        return {'norm_E': actual_norm_E, 'converged': False, 'iterations': 0, 'final_error': float('inf')}
    
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
        
        return {
            'norm_E': actual_norm_E,  # Actual ||E|| value (should equal target_epsilon)
            'converged': bool(converged),
            'iterations': int(iterations + 1),
            'final_error': final_error
        }
    except Exception:
        return {'norm_E': actual_norm_E, 'converged': False, 'iterations': 0, 'final_error': float('inf')}


def test_norm_perturbation_theory(n_matrices=100, matrix_size=50): #maybe print or specifically save the exact final value
    """
    Test the classical theory: if norm(x*P) < 1, convergence is guaranteed.
    
    For each matrix:
    1. Test 5 points below norm(x*P) = 1 (0.2, 0.4, 0.6, 0.8, 0.95)
    2. Test 5 points above norm(x*P) = 1 (1.05, 1.2, 1.4, 1.6, 1.8)
    3. Do binary search (10 iterations) to find exact divergence point
    
    Args:
        n_matrices: Number of random matrices to test
        matrix_size: Size of test matrices
    
    Returns:
        List of dicts with: norm_xP, converged, condition_number, test_type, matrix_idx
    """
    # print("="*60)
    # print("TESTING NORM(x*P) THEORY")
    # print("="*60)
    # print(f"Testing {n_matrices} random {matrix_size}x{matrix_size} matrices")
    # print(f"Total matrices being tested: {n_matrices}")
    # print("For each matrix: 5 below 1, 5 above 1, then binary search for divergence point")
    
    results = []
    
    # Test points: 5 below 1, 5 above 1
    test_points_below = [0.2, 0.4, 0.6, 0.8, 0.95]
    test_points_above = [1.05, 1.2, 1.4, 1.6, 1.8]
    
    for matrix_idx in range(n_matrices):
        # if (matrix_idx + 1) % 5 == 0:
        #     print(f"  Processing matrix {matrix_idx + 1}/{n_matrices}")
        
        # Generate random matrix with controlled condition number
        np.random.seed(matrix_idx)
        # Use wider range of scaling factors to get wider range of condition numbers
        # Smaller scaling_factor → closer to singular → higher condition number
        # Larger scaling_factor → more stable → lower condition number
        # Range from very small (high cond) to larger (low cond)
        scaling_factor = 0.001 + (matrix_idx / n_matrices) * 9.999  # Range: 0.001 to 10.0
        R = np.random.randn(matrix_size, matrix_size) + scaling_factor * np.eye(matrix_size)
        
        # Normalize R to have norm ~1 (as suggested in comments)
        R = R / np.linalg.norm(R)
        
        # Compute condition number AFTER normalization (condition number is scale-invariant, but compute from final R)
        cond_num = float(np.linalg.cond(R))
        
        R_inv = np.linalg.inv(R)
        
        # Generate one random perturbation matrix P for this matrix
        np.random.seed(matrix_idx + 1000)  # Different seed for P
        P = np.random.randn(matrix_size, matrix_size)
        # Normalize P to have norm(P) = 1, so norm(x*P) = x
        P = P / np.linalg.norm(P)
        
        # Compute spectral radius of P * R_inv (for spectral radius theory: ρ(P * A^(-1)) < 1)
        P_times_R_inv = P @ R_inv
        spectral_radius_base = float(np.max(np.abs(np.linalg.eigvals(P_times_R_inv))))
        
        # Test 5 points below ε = 1 (classical threshold: ||E|| < 1 guarantees convergence)
        for epsilon in test_points_below:
            result = test_single_perturbation(R, R_inv, P, epsilon, matrix_size)
            result['condition_number'] = cond_num
            result['test_type'] = 'below_1'
            result['matrix_idx'] = matrix_idx
            # Compute spectral radius: ρ(epsilon * P * R_inv) = epsilon * ρ(P * R_inv)
            result['spectral_radius'] = epsilon * spectral_radius_base
            results.append(result)
        
        # Test 5 points above ε = 1
        for epsilon in test_points_above:
            result = test_single_perturbation(R, R_inv, P, epsilon, matrix_size)
            result['condition_number'] = cond_num
            result['test_type'] = 'above_1'
            result['matrix_idx'] = matrix_idx
            # Compute spectral radius: ρ(epsilon * P * R_inv) = epsilon * ρ(P * R_inv)
            result['spectral_radius'] = epsilon * spectral_radius_base
            results.append(result)
        
        # Binary search to find the exact boundary between converged and diverged
        # Find the rightmost (largest epsilon) that converged and leftmost (smallest epsilon) that diverged
        matrix_results = [r for r in results if r.get('matrix_idx') == matrix_idx]
        converged_points = [r['norm_E'] for r in matrix_results if r['converged']]
        diverged_points = [r['norm_E'] for r in matrix_results if not r['converged']]
        
        if converged_points and diverged_points:
            # Binary search: find boundary between rightmost converged and leftmost diverged
            # low = rightmost (largest) epsilon that converged
            # high = leftmost (smallest) epsilon that diverged
            low = max(converged_points)  # Rightmost converged point
            high = min(diverged_points)   # Leftmost diverged point
            
            # Binary search: repeatedly test the midpoint between low and high
            for bs_iter in range(15):  # Increased iterations for better precision
                mid = (low + high) / 2.0
                result = test_single_perturbation(R, R_inv, P, mid, matrix_size)
                result['condition_number'] = cond_num
                result['test_type'] = 'binary_search'
                result['matrix_idx'] = matrix_idx
                # Compute spectral radius: ρ(epsilon * P * R_inv) = epsilon * ρ(P * R_inv)
                result['spectral_radius'] = mid * spectral_radius_base
                results.append(result)
                
                # Update bounds based on result
                if result['converged']:
                    # If mid converged, the boundary is between mid and high
                    low = mid
                else:
                    # If mid diverged, the boundary is between low and mid
                    high = mid
                
                # Stop if we're close enough (boundary found)
                if abs(high - low) < 1e-8:
                    break
        elif converged_points and not diverged_points:
            # All converged - test smaller epsilon values to find where it might diverge
            # Start from smallest converged point and go smaller
            min_converged = min(converged_points)
            # Test progressively smaller values: 0.1, 0.05, 0.01, 0.005, 0.001
            smaller_tests = [0.1, 0.05, 0.01, 0.005, 0.001]
            for small_eps in smaller_tests:
                if small_eps < min_converged:
                    result = test_single_perturbation(R, R_inv, P, small_eps, matrix_size)
                    result['condition_number'] = cond_num
                    result['test_type'] = 'edge_case_smaller'
                    result['matrix_idx'] = matrix_idx
                    result['spectral_radius'] = small_eps * spectral_radius_base
                    results.append(result)
                    # Stop if we find a divergence
                    if not result['converged']:
                        break
            # Also test larger to see if it eventually diverges
            max_converged = max(converged_points)
            larger_tests = [max_converged * 1.5, max_converged * 2.0, max_converged * 3.0]
            for large_eps in larger_tests:
                result = test_single_perturbation(R, R_inv, P, large_eps, matrix_size)
                result['condition_number'] = cond_num
                result['test_type'] = 'edge_case_larger'
                result['matrix_idx'] = matrix_idx
                result['spectral_radius'] = large_eps * spectral_radius_base
                results.append(result)
                # Stop if we find convergence (unlikely but check)
                if result['converged']:
                    break
        elif diverged_points and not converged_points:
            # All diverged - test smaller epsilon values to find where it might converge
            # Start from smallest diverged point and go smaller
            min_diverged = min(diverged_points)
            # Test progressively smaller values
            smaller_tests = [min_diverged * 0.5, min_diverged * 0.25, min_diverged * 0.1, 0.01, 0.001]
            for small_eps in smaller_tests:
                if small_eps < min_diverged:
                    result = test_single_perturbation(R, R_inv, P, small_eps, matrix_size)
                    result['condition_number'] = cond_num
                    result['test_type'] = 'edge_case_smaller'
                    result['matrix_idx'] = matrix_idx
                    result['spectral_radius'] = small_eps * spectral_radius_base
                    results.append(result)
                    # Stop if we find convergence
                    if result['converged']:
                        break
    
    return results


def plot_perturbation_results(perturbation_results, norm_theory_results=None):
    """
    Plot the results of perturbation robustness testing.
    
    Creates 3 plots:
    1. Threshold vs condition number: Does robustness depend on condition number?
    2. Steps to divergence: How many iterations until divergence?
    3. Norm(x*P) theory: Testing if norm(x*P) < 1 guarantees convergence
    
    Args:
        perturbation_results: List of dicts from test_perturbation_robustness()
        norm_theory_results: List of dicts from test_norm_perturbation_theory() (optional)
    
    Returns:
        None (displays plots)
    """
    if not perturbation_results:
        # print("No perturbation results to plot")
        return
    
    # Create 4 plots if norm_theory_results provided, otherwise 2
    if norm_theory_results:
        fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = [axes[0], axes[1], None, None]
    
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
    
    axes[0].scatter(cond_nums, thresholds, alpha=0.5, s=30, edgecolors='black', linewidths=0.3)
    axes[0].set_xlabel('Condition Number', fontsize=11)
    axes[0].set_ylabel('Max Perturbation Before Divergence', fontsize=11)
    axes[0].set_title('Robustness vs Condition Number', fontsize=12)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Steps to divergence vs perturbation
    axes[1].set_xlabel('Perturbation Size', fontsize=11)
    axes[1].set_ylabel('Steps to Divergence', fontsize=11)
    axes[1].set_title('Steps to Divergence vs Perturbation', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    for i, result in enumerate(perturbation_results[:25]):  # Show more matrices
        perturbations = [d['perturbation'] for d in result['convergence_data']]
        iterations = [d['iterations'] for d in result['convergence_data']]
        converged_flags = [d['converged'] for d in result['convergence_data']]
        
        # Plot diverged cases: show how many steps it took to diverge
        diverged_x = [p for p, c in zip(perturbations, converged_flags) if not c]
        diverged_steps = [it for it, c in zip(iterations, converged_flags) if not c]
        
        if diverged_x and diverged_steps:
            axes[1].plot(diverged_x, diverged_steps, 'o-', color='red', 
                            alpha=0.3, markersize=3, linewidth=1.0)
    
    # Plot 3: Norm(x*P) theory test
    # magnitude of x * P, classical theory: if norm(x*P)<1, the you are GAURANTEED to converge; any orthogonal matrix has cond 1, not just I. maybe try to test
    # look at stats of random matrices not the individual matrix to find trend
    # many matries of same cond and random perturbation
    # uniform distribution might have bad cond
    # normally distributed, sqrt n is approx norm
    # how to pick othorgonal matrix correctly
    if norm_theory_results:
        # Count unique matrices
        unique_matrices = set(r.get('matrix_idx', -1) for r in norm_theory_results)
        n_matrices_tested = len([m for m in unique_matrices if m >= 0])
        
        # Group results by matrix_idx to plot horizontal lines
        # Each matrix has ONE condition number (from original R) and ~20 different norm(x*P) tests
        # X-axis: norm(x*P) (perturbation scale) - varies for each test
        # Y-axis: condition number (constant for each matrix) - creates horizontal lines
        matrices_dict = {}
        for r in norm_theory_results:
            matrix_idx = r.get('matrix_idx', -1)
            if matrix_idx >= 0:
                if matrix_idx not in matrices_dict:
                    # Store condition number from original matrix (same for all tests of this matrix)
                    matrices_dict[matrix_idx] = {
                        'condition_number': r['condition_number'],  # From original R, constant for all tests
                        'norm_E': [],  # ||E|| values (ε in classical theory)
                        'converged': []
                    }
                # Add this test's ||E|| value (varies for each test, this is ε in classical theory)
                matrices_dict[matrix_idx]['norm_E'].append(r['norm_E'])
                matrices_dict[matrix_idx]['converged'].append(r['converged'])
        
        # Plot each matrix as a horizontal line (same condition number, different norm(x*P) values)
        # Sort by matrix_idx to ensure consistent plotting
        sorted_matrix_indices = sorted(matrices_dict.keys())
        
        # Verify we have the expected number of matrices and data
        # print(f"DEBUG: Found {len(sorted_matrix_indices)} unique matrices in results")
        # print(f"DEBUG: Total test results: {len(norm_theory_results)}")
        # if len(sorted_matrix_indices) > 0:
        #     sample_matrix = matrices_dict[sorted_matrix_indices[0]]
        #     print(f"DEBUG: Sample matrix has {len(sample_matrix['norm_E'])} tests, cond_num={sample_matrix['condition_number']:.6e}")
        
        for matrix_idx in sorted_matrix_indices:
            data = matrices_dict[matrix_idx]
            cond_num = data['condition_number']  # Same for all tests of this matrix
            norm_E_vals = data['norm_E']  # ||E|| values (ε in classical theory, different for each test)
            converged_flags = data['converged']
            
            # Sort by ||E|| to make lines clearer
            sorted_data = sorted(zip(norm_E_vals, converged_flags), key=lambda x: x[0])
            norm_E_vals = [x[0] for x in sorted_data]
            converged_flags = [x[1] for x in sorted_data]
            
            # Separate converged and diverged points for this matrix
            converged_norms = [n for n, c in zip(norm_E_vals, converged_flags) if c]
            diverged_norms = [n for n, c in zip(norm_E_vals, converged_flags) if not c]
            
            # Plot converged points in green - all at same Y (cond_num), different X (norm_xP)
            # Use line plot to make horizontal lines visible
            if len(converged_norms) > 1:
                axes[2].plot(converged_norms, [cond_num]*len(converged_norms), 
                           color='green', alpha=0.4, linewidth=0.8, marker='o', 
                           markersize=3, zorder=2, label='Converged' if matrix_idx == 0 else '')
            elif len(converged_norms) == 1:
                axes[2].scatter(converged_norms, [cond_num], 
                              color='green', alpha=0.4, s=20, zorder=2, marker='o',
                              label='Converged' if matrix_idx == 0 else '')
            
            # Plot diverged points in red - all at same Y (cond_num), different X (norm_xP)
            if len(diverged_norms) > 1:
                axes[2].plot(diverged_norms, [cond_num]*len(diverged_norms), 
                           color='red', alpha=0.4, linewidth=0.8, marker='o', 
                           markersize=3, zorder=2, label='Diverged' if matrix_idx == 0 else '')
            elif len(diverged_norms) == 1:
                axes[2].scatter(diverged_norms, [cond_num], 
                              color='red', alpha=0.4, s=20, zorder=2, marker='o',
                              label='Diverged' if matrix_idx == 0 else '')
        
        # Add vertical line at ||E|| = 1 (classical threshold: ||E|| < 1 guarantees convergence)
        axes[2].axvline(x=1.0, color='black', linestyle='--', linewidth=2.0, 
                       label='||E|| = 1 (ε = 1)', zorder=1)
        
        axes[2].set_xlabel('||E|| (Perturbation Norm, ε)', fontsize=11)
        axes[2].set_ylabel('Condition Number', fontsize=11)
        axes[2].set_title(f'Norm(x*P) Theory Test\n({n_matrices_tested} matrices)', fontsize=12)
        axes[2].set_yscale('log')
        # Set y-axis limits based on data range (log scale uses multiplicative padding)
        if matrices_dict:
            all_cond_nums = [data['condition_number'] for data in matrices_dict.values()]
            y_min = min(all_cond_nums)
            y_max = max(all_cond_nums)
            # For log scale, use multiplicative padding (not additive)
            axes[2].set_ylim([y_min * 0.5, y_max * 2.0])
        axes[2].grid(True, alpha=0.3, zorder=0)
        axes[2].legend(loc='best', fontsize=9)
        
        # Plot 4: Spectral radius theory: ρ(ε·P·A⁻¹) < 1
        # Same structure as Plot 3, but x-axis is spectral radius instead of ||E||
        # Group results by matrix_idx to plot horizontal lines
        matrices_dict_spec = {}
        for r in norm_theory_results:
            matrix_idx = r.get('matrix_idx', -1)
            if matrix_idx >= 0 and 'spectral_radius' in r:
                if matrix_idx not in matrices_dict_spec:
                    matrices_dict_spec[matrix_idx] = {
                        'condition_number': r['condition_number'],
                        'spectral_radius': [],
                        'converged': []
                    }
                matrices_dict_spec[matrix_idx]['spectral_radius'].append(r['spectral_radius'])
                matrices_dict_spec[matrix_idx]['converged'].append(r['converged'])
        
        # Plot each matrix as a horizontal line (same structure as Plot 3)
        sorted_matrix_indices_spec = sorted(matrices_dict_spec.keys())
        for matrix_idx in sorted_matrix_indices_spec:
            data = matrices_dict_spec[matrix_idx]
            cond_num = data['condition_number']  # Same for all tests of this matrix
            spec_rad_vals = data['spectral_radius']  # Different for each test
            converged_flags = data['converged']
            
            # Sort by spectral radius to make lines clearer
            sorted_data = sorted(zip(spec_rad_vals, converged_flags), key=lambda x: x[0])
            spec_rad_vals = [x[0] for x in sorted_data]
            converged_flags = [x[1] for x in sorted_data]
            
            # Separate converged and diverged points for this matrix
            converged_spec = [s for s, c in zip(spec_rad_vals, converged_flags) if c]
            diverged_spec = [s for s, c in zip(spec_rad_vals, converged_flags) if not c]
            
            # Plot converged points in green - all at same Y (cond_num), different X (spectral_radius)
            if len(converged_spec) > 1:
                axes[3].plot(converged_spec, [cond_num]*len(converged_spec), 
                           color='green', alpha=0.4, linewidth=0.8, marker='o', 
                           markersize=3, zorder=2, label='Converged' if matrix_idx == 0 else '')
            elif len(converged_spec) == 1:
                axes[3].scatter(converged_spec, [cond_num], 
                              color='green', alpha=0.4, s=20, zorder=2, marker='o',
                              label='Converged' if matrix_idx == 0 else '')
            
            # Plot diverged points in red - all at same Y (cond_num), different X (spectral_radius)
            if len(diverged_spec) > 1:
                axes[3].plot(diverged_spec, [cond_num]*len(diverged_spec), 
                           color='red', alpha=0.4, linewidth=0.8, marker='o', 
                           markersize=3, zorder=2, label='Diverged' if matrix_idx == 0 else '')
            elif len(diverged_spec) == 1:
                axes[3].scatter(diverged_spec, [cond_num], 
                              color='red', alpha=0.4, s=20, zorder=2, marker='o',
                              label='Diverged' if matrix_idx == 0 else '')
        
        # Add vertical line at spectral radius = 1
        axes[3].axvline(x=1.0, color='black', linestyle='--', linewidth=2.0, 
                       label='ρ(ε·P·A⁻¹) = 1', zorder=1)
        
        axes[3].set_xlabel('Spectral Radius ρ(ε·P·A⁻¹)', fontsize=11)
        axes[3].set_ylabel('Condition Number', fontsize=11)
        axes[3].set_title(f'Spectral Radius Theory\n({n_matrices_tested} matrices)', fontsize=12)
        axes[3].set_yscale('log')
        # Set x-axis limits to center around 1, cut off outliers
        # Focus on range 0 to 2 to see behavior around threshold
        axes[3].set_xlim([0, 2.0])
        if matrices_dict_spec:
            all_cond_nums_spec = [data['condition_number'] for data in matrices_dict_spec.values()]
            y_min_spec = min(all_cond_nums_spec)
            y_max_spec = max(all_cond_nums_spec)
            axes[3].set_ylim([y_min_spec * 0.5, y_max_spec * 2.0])
        axes[3].grid(True, alpha=0.3, zorder=0)
        axes[3].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    # thresholds_float = [float(t) for t in thresholds]
    # print("\n" + "="*60)
    # print("PERTURBATION ROBUSTNESS SUMMARY")
    # print("="*60)
    # print(f"Number of matrices tested: {int(len(perturbation_results))}")
    # print(f"Average convergence threshold: {float(np.mean(thresholds_float)):.6f}")
    # print(f"Median convergence threshold: {float(np.median(thresholds_float)):.6f}")
    # print(f"Min threshold: {float(np.min(thresholds_float)):.6f}")
    # print(f"Max threshold: {float(np.max(thresholds_float)):.6f}")
    # print(f"Std deviation: {float(np.std(thresholds_float)):.6f}")


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
    
    # Run the norm(x*P) theory test
    norm_theory_results = test_norm_perturbation_theory(
        n_matrices=100,              # Test 100 random matrices
        matrix_size=50              # 50x50 matrices
    )
    
    # Plot the results
    plot_perturbation_results(perturbation_results, norm_theory_results)

