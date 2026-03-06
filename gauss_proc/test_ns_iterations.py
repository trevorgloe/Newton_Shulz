"""
Test script: Compare Newton-Schulz iteration counts between:
  - Cold start: scaled Khat as initial guess (baseline)
  - Warm start: inverse from previous GD iteration as initial guess

Runs multiple GP trials with different random seeds and produces comparison plots.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import contextlib

# --- Config ---
N_TRIALS = 5
N_POINTS = 20
SIG = 0.01
ALPHA = 1e-6
MAX_ITER = 60
EPS = 1e-5

def run_single_trial(seed: int, use_prev_inv_init: bool):
    """Run one GP grad descent with given seed and init strategy. Returns (theta_hist, ns_iterations)."""
    np.random.seed(seed)
    n = N_POINTS
    sig = SIG
    x = np.random.rand(n)
    y = np.sin(x * 2 * np.pi) + sig * np.random.randn(n)

    theta0 = np.array([0.1, 0.01])

    gp = GP.GPR(n, 1, "RBF", sig, use_ns=True, use_prev_inv_init=use_prev_inv_init, verbose=False)
    gp.fit(np.array([x]), y, np.array([sig * 10, 1e-2]))

    # with contextlib.redirect_stdout(io.StringIO()):
    result = gp.grad_dec_theta(theta0, alpha=ALPHA, max_iter=MAX_ITER, eps=EPS,
                                  return_theta=True, return_ns_iterations=True)
    theta_hist, ns_iters = result
    return theta_hist, ns_iters


def main():
    print("=" * 60)
    print("NS Iteration Comparison: Cold Start vs Warm Start (Prev Inv Init)")
    print("=" * 60)
    print(f"Running {N_TRIALS} trials with n={N_POINTS}, alpha={ALPHA}, max_iter={MAX_ITER}")
    print()

    cold_iters_all = []
    warm_iters_all = []
    cold_totals = []
    warm_totals = []
    cold_n_calls = []
    warm_n_calls = []
    cold_per_gd = []
    warm_per_gd = []

    for trial in range(N_TRIALS):
        seed = trial
        print(f"Trial {trial + 1}/{N_TRIALS} (seed={seed})...")

        # Cold start (scaled Khat as init each time)
        print("Cold start")
        theta_c, ns_c = run_single_trial(seed, use_prev_inv_init=False)
        cold_iters_all.append(ns_c)
        cold_totals.append(sum(ns_c))
        cold_n_calls.append(len(ns_c))
        cold_per_gd.append(ns_c)

        # Warm start (prev inverse as init)
        print("Warm start")
        theta_w, ns_w = run_single_trial(seed, use_prev_inv_init=True)
        warm_iters_all.append(ns_w)
        warm_totals.append(sum(ns_w))
        warm_n_calls.append(len(ns_w))
        warm_per_gd.append(ns_w)

        print(f"  Cold: total NS iters = {sum(ns_c)}, # NS calls = {len(ns_c)}")
        print(f"  Warm: total NS iters = {sum(ns_w)}, # NS calls = {len(ns_w)}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(N_TRIALS)
    width = 0.35

    # Left: Total NS iterations per trial
    ax1.bar(x_pos - width/2, cold_totals, width, label='Cold start (scaled Khat init)', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, warm_totals, width, label='Warm start (prev inv init)', color='coral', alpha=0.8)
    ax1.set_xlabel('Trial (seed)')
    ax1.set_ylabel('Total NS iterations')
    ax1.set_title('Total Newton-Schulz Iterations per Trial')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(i) for i in range(N_TRIALS)])
    ax1.legend()

    # Right: # of NS calls per trial
    ax2.bar(x_pos - width/2, cold_n_calls, width, label='Cold start', color='steelblue', alpha=0.8)
    ax2.bar(x_pos + width/2, warm_n_calls, width, label='Warm start', color='coral', alpha=0.8)
    ax2.set_xlabel('Trial (seed)')
    ax2.set_ylabel('# of NS calls')
    ax2.set_title('# of NS Calls per Trial')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(i) for i in range(N_TRIALS)])
    ax2.legend()

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'ns_iteration_comparison.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"\nSaved {outpath}")

    print("\n" + "=" * 60)
    print("Process Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
