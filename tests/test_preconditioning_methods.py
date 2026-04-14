"""
Test and compare preconditioning methods for Newton-Schulz matrix inversion.

This script compares:
1. No preconditioning
2. ILU (Incomplete LU)
3. prrLU-style partial pivoted LU

Convergence is measured with ||A @ G - I|| at each iteration.

Important initial-guess modes:
- initial_guess_mode = 0:
    Use a cheap generic guess G0 = alpha * A_used^T, where A_used is the
    matrix actually iterated on. For preconditioned methods that means the
    preconditioned matrix, not the original matrix.
- initial_guess_mode = 1:
    Use the exact inverse of the unperturbed reference system as the starting
    guess, with no added perturbation. For preconditioned methods this means:
        (1) factor/precondition the perturbed test matrix as usual,
        (2) apply those SAME left/right transformation operators to the saved
            unperturbed matrix,
        (3) invert that transformed unperturbed matrix,
        (4) use that inverse directly as the starting guess.
- initial_guess_mode = 2:
    Same as mode 1, except after building that reference inverse guess we add a
    Gaussian perturbation using the SAME global matrix perturbation scale. This
    restores the earlier perturbation style without needing a separate initial-
    guess perturbation scale.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Dict, Any, Callable
from pathlib import Path
from matplotlib.gridspec import GridSpec

import numpy as np

# Matplotlib backend handling:
# - In headless runs (common in CI / remote), MPLBACKEND is often "Agg".
# - Don't attempt to force a GUI backend in that case (can crash).
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import preconditioners as ilu


ArrayLike = np.ndarray

# Global Test Settings
DEFAULT_MAX_ITERS = 15
DEFAULT_MATRIX_SIZE = 50
DEFAULT_N_TRIALS = 50
DEFAULT_SEED_BASE = 2
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_EXPLOSION_THRESHOLD = 1e10
DEFAULT_PERTURBATION_SCALE = 1e-2
DEFAULT_INITIAL_GUESS_SCALER = 1e-3
DEFAULT_MAX_MATRIX_RETRIES = 100


# =============================================================================
# Helper Methods
# =============================================================================

def _inverse_or_none(A: ArrayLike) -> ArrayLike | None:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None

def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _apply_gaussian_perturbation(A: ArrayLike, seed: int) -> ArrayLike:
    rng = _make_rng(seed)
    noise = rng.standard_normal(A.shape)
    return A + DEFAULT_PERTURBATION_SCALE * noise

def _add_guess_perturbation(G: ArrayLike, seed: int) -> ArrayLike:
    rng = _make_rng(seed)
    noise = rng.standard_normal(G.shape)
    return G + DEFAULT_PERTURBATION_SCALE * noise

def _generate_invertible_matrix_with_perturbation(
    n: int,
    seed: int,
    max_retries: int = DEFAULT_MAX_MATRIX_RETRIES,
) -> Tuple[ArrayLike, ArrayLike, List[Dict[str, ArrayLike]], int]:
    """
    Generate:
      - A_original: unperturbed invertible matrix (randn with seed)
      - A: perturbed invertible matrix
      - matrix_array: array storing original unperturbed matrix and its inverse,
        and also the perturbed matrix and its inverse
      - used_seed: actual seed used on successful generation

    If either original or perturbed matrix is singular, regenerate with a new seed.
    """
    for retry in range(max_retries):
        used_seed = seed + retry
        rng = _make_rng(used_seed)
        A_original = rng.standard_normal((n, n))
        A_original_inv = _inverse_or_none(A_original)
        if A_original_inv is None:
            continue

        A = _apply_gaussian_perturbation(A_original, used_seed + 1)
        A_inv = _inverse_or_none(A)
        if A_inv is None:
            continue

        matrix_array = [
            {
                "matrix": A_original,
                "inverse": A_original_inv,
            },
            {
                "matrix": A,
                "inverse": A_inv,
            },
        ]
        return A_original, A, matrix_array, used_seed

    raise ValueError(
        f"Failed to generate invertible original/perturbed matrices after {max_retries} retries."
    )


def _safe_condition_number(A: ArrayLike) -> float:
    try:
        return float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")

def _is_exploded(M: ArrayLike, explosion_threshold: float) -> bool:
    return (
        np.any(np.isnan(M))
        or np.any(np.isinf(M))
        or np.any(np.abs(M) > explosion_threshold)
    )

def _newton_schulz_step(A: ArrayLike, G: ArrayLike) -> ArrayLike:
    # Equivalent to G_{k+1} = G_k (2I - A G_k) = 2G - G A G
    return G + G @ (np.eye(A.shape[0]) - A @ G)


def _run_newton_schulz_core(
    A_iter: ArrayLike,
    G0: ArrayLike,
    *,
    A_error: ArrayLike | None = None,
    G_map: Callable[[ArrayLike], ArrayLike] | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    explosion_threshold: float = DEFAULT_EXPLOSION_THRESHOLD,
) -> Tuple[ArrayLike, List[float], int, str]:
    """
    Iterate Newton-Schulz on A_iter using initial guess G0.

    A_error and G_map allow us to iterate on a preconditioned system while still
    reporting error on the original matrix/inverse pair.
    """
    if A_error is None:
        A_error = A_iter
    if G_map is None:
        G_map = lambda G: G

    I_err = np.eye(A_error.shape[0])
    G_iter = G0.copy()
    G_eval = G_map(G_iter)
    initial_error = float(np.linalg.norm(A_error @ G_eval - I_err))
    errors = [initial_error]

    if not np.isfinite(initial_error):
        return G_eval, errors, 0, "invalid_initial_error"
    if initial_error < convergence_threshold:
        return G_eval, errors, 0, "converged"

    status = "max_iters"
    for _ in range(max_iters):
        G_new = _newton_schulz_step(A_iter, G_iter)
        if _is_exploded(G_new, explosion_threshold):
            status = "explosion"
            break

        G_iter = G_new
        G_eval = G_map(G_iter)
        error = float(np.linalg.norm(A_error @ G_eval - I_err))
        errors.append(error)

        if not np.isfinite(error):
            status = "invalid_error"
            break
        if error < convergence_threshold:
            status = "converged"
            break

    return G_eval, errors, len(errors) - 1, status


def _validate_initial_guess_mode(initial_guess_mode: int) -> None:
    if initial_guess_mode not in (0, 1, 2):
        raise ValueError("initial_guess_mode must be 0, 1, or 2")


def _build_no_precond_initial_guess(
    A: ArrayLike,
    A_unperturbed: ArrayLike | None,
    *,
    initial_guess_mode: int,
    perturb_seed: int = 0,
) -> ArrayLike:
    _validate_initial_guess_mode(initial_guess_mode)

    if initial_guess_mode == 0:
        alpha = 1.0 / (np.linalg.norm(A, 1) * np.linalg.norm(A, np.inf))
        return alpha * A.T

    if A_unperturbed is None:
        raise ValueError("A_unperturbed must be provided when initial_guess_mode is 1 or 2")

    G0 = _inverse_or_none(A_unperturbed)
    if G0 is None:
        raise ValueError("A_unperturbed must be invertible when initial_guess_mode is 1 or 2")

    if initial_guess_mode == 2:
        return _add_guess_perturbation(G0, perturb_seed)

    return G0


def _build_precond_initial_guess(
    A_precond: ArrayLike,
    A_unperturbed: ArrayLike | None,
    L: ArrayLike,
    U: ArrayLike,
    P: ArrayLike | None = None,
    Q: ArrayLike | None = None,
    *,
    initial_guess_mode: int,
    perturb_seed: int,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    _validate_initial_guess_mode(initial_guess_mode)

    if initial_guess_mode == 0:
        alpha = 1.0 / (np.linalg.norm(A_precond, 1) * np.linalg.norm(A_precond, np.inf))
        return alpha * A_precond.T, {
            "initial_guess_description": "A_precond.T / (||A_precond||_1 ||A_precond||_inf)"
        }

    if A_unperturbed is None:
        raise ValueError("A_unperturbed must be provided when initial_guess_mode is 1 or 2")

    A_unperturbed_transformed = ilu.precondition_matrix(A_unperturbed, L, U, P, Q)
    G0_base = _inverse_or_none(A_unperturbed_transformed)
    if G0_base is None:
        raise ValueError(
            "The transformed unperturbed matrix was singular in initial_guess_mode 1/2"
        )

    if initial_guess_mode == 2:
        G0 = _add_guess_perturbation(G0_base, perturb_seed)
        desc = (
            "inv(transformed unperturbed matrix using perturbed-matrix operators) "
            "+ Gaussian perturbation scaled by DEFAULT_PERTURBATION_SCALE"
        )
    else:
        G0 = G0_base
        desc = "inv(transformed unperturbed matrix using perturbed-matrix operators)"

    return G0, {
        "initial_guess_description": desc,
        "cond_A_unperturbed_transformed": _safe_condition_number(A_unperturbed_transformed),
    }


# =============================================================================
# Newton-Schulz with Preconditioning
# =============================================================================

def newton_schulz_no_precond(
    A: ArrayLike,
    max_iters: int = DEFAULT_MAX_ITERS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    explosion_threshold: float = DEFAULT_EXPLOSION_THRESHOLD,
    initial_guess_mode: int = 0,
    A_unperturbed: ArrayLike | None = None,
    initial_guess_seed: int = 0,
) -> Tuple[ArrayLike, List[float], int, Dict[str, Any]]:
    """
    Run Newton-Schulz without preconditioning.

    initial_guess_mode:
      0 -> G0 = alpha * A^T
      1 -> G0 = inv(A_unperturbed)
      2 -> G0 = inv(A_unperturbed) + Gaussian perturbation
    """
    cond_A = _safe_condition_number(A)
    G0 = _build_no_precond_initial_guess(
        A,
        A_unperturbed,
        initial_guess_mode=initial_guess_mode,
        perturb_seed=initial_guess_seed,
    )

    G, errors, iters, status = _run_newton_schulz_core(
        A,
        G0,
        max_iters=max_iters,
        convergence_threshold=convergence_threshold,
        explosion_threshold=explosion_threshold,
    )

    info = {
        "cond_A": cond_A,
        "status": status,
        "initial_guess_mode": initial_guess_mode,
        "initial_guess_norm": float(np.linalg.norm(G0)),
        "initial_guess_seed": int(initial_guess_seed),
    }
    return G, errors, iters, info


def newton_schulz_with_ilu_precond(
    A: ArrayLike,
    max_iters: int = DEFAULT_MAX_ITERS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    explosion_threshold: float = DEFAULT_EXPLOSION_THRESHOLD,
    drop_tol: float = 1e-3,
    initial_guess_mode: int = 0,
    A_unperturbed: ArrayLike | None = None,
    initial_guess_seed: int = 0,
) -> Tuple[ArrayLike, List[float], int, Dict[str, Any]]:
    """
    Run Newton-Schulz with ILU preconditioning.

    Newton-Schulz is iterated on the preconditioned matrix. Error is still
    reported on the recovered inverse for the original matrix.
    """
    L, U = ilu.incomplete_lu(A, drop_tol=drop_tol)
    A_precond = ilu.precondition_matrix(A, L, U)

    cond_A = _safe_condition_number(A)
    cond_A_precond = _safe_condition_number(A_precond)

    G_p0, extra_init_info = _build_precond_initial_guess(
        A_precond,
        A_unperturbed,
        L,
        U,
        None,
        None,
        initial_guess_mode=initial_guess_mode,
        perturb_seed=initial_guess_seed,
    )

    G, errors, iters, status = _run_newton_schulz_core(
        A_precond,
        G_p0,
        A_error=A,
        G_map=lambda G_p: ilu.recover_inverse(G_p, L, U),
        max_iters=max_iters,
        convergence_threshold=convergence_threshold,
        explosion_threshold=explosion_threshold,
    )

    info = {
        "cond_A": cond_A,
        "cond_A_precond": cond_A_precond,
        "L": L,
        "U": U,
        "status": status,
        "initial_guess_mode": initial_guess_mode,
        "initial_guess_norm": float(np.linalg.norm(G_p0)),
        **extra_init_info,
    }
    return G, errors, iters, info


def newton_schulz_with_prrlu_precond(
    A: ArrayLike,
    max_iters: int = DEFAULT_MAX_ITERS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    explosion_threshold: float = DEFAULT_EXPLOSION_THRESHOLD,
    num_rows: int | None = None,
    initial_guess_mode: int = 0,
    A_unperturbed: ArrayLike | None = None,
    initial_guess_seed: int = 0,
) -> Tuple[ArrayLike, List[float], int, Dict[str, Any]]:
    """
    Run Newton-Schulz with prrLU preconditioning.

    Newton-Schulz is iterated on the preconditioned matrix. Error is still
    reported on the recovered inverse for the original matrix.
    """
    n = A.shape[0]
    if num_rows is None:
        num_rows = n // 2
    num_rows = max(1, min(num_rows, n))

    L, U, P, Q = ilu.prrlu(A, num_rows=num_rows)
    A_precond = ilu.precondition_matrix(A, L, U, P, Q)

    cond_A = _safe_condition_number(A)
    cond_A_precond = _safe_condition_number(A_precond)

    G_p0, extra_init_info = _build_precond_initial_guess(
        A_precond,
        A_unperturbed,
        L,
        U,
        P,
        Q,
        initial_guess_mode=initial_guess_mode,
        perturb_seed=initial_guess_seed,
    )

    G, errors, iters, status = _run_newton_schulz_core(
        A_precond,
        G_p0,
        A_error=A,
        G_map=lambda G_p: ilu.recover_inverse(G_p, L, U, P, Q),
        max_iters=max_iters,
        convergence_threshold=convergence_threshold,
        explosion_threshold=explosion_threshold,
    )

    info = {
        "cond_A": cond_A,
        "cond_A_precond": cond_A_precond,
        "num_rows": num_rows,
        "L": L,
        "U": U,
        "P": P,
        "Q": Q,
        "status": status,
        "initial_guess_mode": initial_guess_mode,
        "initial_guess_norm": float(np.linalg.norm(G_p0)),
        **extra_init_info,
    }
    return G, errors, iters, info


# =============================================================================
# Plotting Functions
# =============================================================================


def _compute_method_summary(detailed_results: List[Dict[str, Any]], method_name: str) -> Dict[str, float]:
    runs = [r for r in detailed_results if r.get("method") == method_name]
    total = len(runs)

    if total == 0:
        return {
            "avg_iters": np.nan,
            "pct_converged": np.nan,
            "pct_exploded": np.nan,
            "pct_iter_limit": np.nan,
            "pct_other": np.nan,
            "median_final_error_converged": np.nan,
            "n_runs": 0,
        }

    converged = [r for r in runs if r.get("status") == "converged"]
    exploded = [r for r in runs if r.get("status") == "explosion"]
    iter_limit = [r for r in runs if r.get("status") == "max_iters"]
    other = [r for r in runs if r.get("status") not in {"converged", "explosion", "max_iters"}]

    converged_iters = [
        r.get("iterations", np.nan)
        for r in converged
        if np.isfinite(r.get("iterations", np.nan))
    ]
    conv_final_errors = [
        r.get("final_error", np.nan)
        for r in converged
        if np.isfinite(r.get("final_error", np.nan))
    ]

    return {
        "avg_iters": float(np.mean(converged_iters)) if converged_iters else np.nan,
        "pct_converged": 100.0 * len(converged) / total,
        "pct_exploded": 100.0 * len(exploded) / total,
        "pct_iter_limit": 100.0 * len(iter_limit) / total,
        "pct_other": 100.0 * len(other) / total,
        "median_final_error_converged": float(np.median(conv_final_errors)) if conv_final_errors else np.nan,
        "n_runs": total,
    }


def _flatten_detailed_results(all_detailed_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []

    for matrix_name, detail in all_detailed_results.items():
        for trial_idx, trial in enumerate(detail.get("trials", [])):
            for key, method_short in [
                ("No Precond", "no_precond"),
                ("ILU Precond", "ilu"),
            ]:
                data = trial[key]
                cond = trial["cond_A"] if method_short == "no_precond" else data.get("cond_precond", np.nan)
                flattened.append({
                    "matrix_name": matrix_name,
                    "trial_idx": trial_idx,
                    "method": method_short,
                    "iterations": data["iterations"],
                    "final_error": data["final_error"],
                    "status": data["status"],
                    "errors": data["errors"],
                    "cond": cond,
                })

            prrlu_key = next(k for k in trial.keys() if "prrLU Precond" in k)
            data = trial[prrlu_key]
            flattened.append({
                "matrix_name": matrix_name,
                "trial_idx": trial_idx,
                "method": "prrlu",
                "iterations": data["iterations"],
                "final_error": data["final_error"],
                "status": data["status"],
                "errors": data["errors"],
                "cond": data.get("cond_precond", np.nan),
            })

    return flattened


def _make_summary_table_rows(detailed_results: List[Dict[str, Any]]) -> List[List[str]]:
    methods = ["no_precond", "ilu", "prrlu"]
    pretty = {
        "no_precond": "No Precond",
        "ilu": "ILU",
        "prrlu": "prrLU",
    }

    rows: List[List[str]] = []
    for m in methods:
        s = _compute_method_summary(detailed_results, m)
        rows.append([
            pretty[m],
            f"{s['avg_iters']:.3f}" if np.isfinite(s["avg_iters"]) else "nan",
            f"{s['pct_converged']:.1f}%",
            f"{s['pct_exploded']:.1f}%",
            f"{s['pct_iter_limit']:.1f}%",
            f"{s['pct_other']:.1f}%",
            f"{s['median_final_error_converged']:.2e}"
            if np.isfinite(s["median_final_error_converged"])
            else "nan",
        ])
    return rows


def plot_aggregated_comparison(
    all_detailed_results: Dict[str, Any],
    save_dir: str | None = None,
) -> None:

    flattened = _flatten_detailed_results(all_detailed_results)
    if not flattened:
        print("No detailed results available; skipping plots.")
        return

    style_map = {
        "no_precond": {"label": "No Precond", "marker": "o", "color": "C0"},
        "ilu": {"label": "ILU", "marker": "s", "color": "C1"},
        "prrlu": {"label": "prrLU", "marker": "^", "color": "C2"},
    }

    AXIS_SIZE = 14
    TITLE_SIZE = 16
    LEGEND_SIZE = 12

    def style_axes(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=AXIS_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=AXIS_SIZE)
        ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------
    # Plot 1 — Iterations vs Trial
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10,7))

    for method in ["no_precond","ilu","prrlu"]:
        runs = [
            r for r in flattened
            if r["method"] == method and r["status"] == "converged"
        ]
        if not runs:
            continue

        style = style_map[method]

        x = [r["trial_idx"] for r in runs]
        y = [r["iterations"] for r in runs]

        ax1.scatter(
            x,
            y,
            label=style["label"],
            marker=style["marker"],
            s=45,
            alpha=0.6,
            color=style["color"],
        )

        avg = np.mean(y)
        ax1.axhline(avg, linestyle="--", color=style["color"], linewidth=1.5)

    style_axes(ax1,"Trial","Iterations","Iterations vs Trial")
    ax1.legend(fontsize=LEGEND_SIZE)

    fig1.tight_layout()

    if save_dir:
        fig1.savefig(os.path.join(save_dir,"iterations_vs_trial.png"),dpi=200)

    # ------------------------------------------------------------
    # Plot 2 — Convergence Curves
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10,7))

    for method in ["no_precond","ilu","prrlu"]:

        runs = [
            r for r in flattened
            if r["method"] == method and r["status"] == "converged"
        ]

        if not runs:
            continue

        style = style_map[method]

        max_traces = 20
        step = max(1,len(runs)//max_traces)
        subset = runs[::step][:max_traces]

        for i,r in enumerate(subset):

            errs = r["errors"]

            ax2.plot(
                range(len(errs)),
                errs,
                linewidth=2,
                alpha=0.2,
                color=style["color"],
                label=style["label"] if i==0 else None,
            )

    ax2.set_yscale("log")
    style_axes(ax2,"Iteration",r"$||AG-I||$","Convergence Curves")
    ax2.legend(fontsize=LEGEND_SIZE)

    fig2.tight_layout()

    if save_dir:
        fig2.savefig(os.path.join(save_dir,"convergence_curves.png"),dpi=200)

    # ------------------------------------------------------------
    # Plot 3 — Condition Numbers
    # ------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10,7))

    for method in ["no_precond","ilu","prrlu"]:

        runs = [r for r in flattened if r["method"] == method]

        if not runs:
            continue

        style = style_map[method]

        conds = [r["cond"] for r in runs if r.get("cond") is not None and np.isfinite(r["cond"])]

        ax3.scatter(
            range(len(conds)),
            conds,
            label=style["label"],
            marker=style["marker"],
            s=45,
            alpha=0.6,
            color=style["color"],
        )

    ax3.set_yscale("log")

    style_axes(ax3, "Trial", "Condition Number", "Condition Numbers")
    ax3.legend(fontsize=LEGEND_SIZE)

    fig3.tight_layout()

    if save_dir:
        fig3.savefig(os.path.join(save_dir, "condition_numbers.png"), dpi=200)

    plt.show()

# =============================================================================
# Main Testing Functions
# =============================================================================

def run_single_matrix_test(
    A: ArrayLike,
    matrix_name: str = "Test Matrix",
    max_iters: int = DEFAULT_MAX_ITERS,
    prrlu_rows: int | None = None,
    initial_guess_mode: int = 0,
    A_unperturbed: ArrayLike | None = None,
    initial_guess_seed: int = 0,
) -> Dict[str, Any]:
    n = A.shape[0]
    cond_A = _safe_condition_number(A)

    if A_unperturbed is None:
        A_unperturbed = A

    print(f"\n{'='*60}")
    print(f"Testing: {matrix_name}")
    print(f"Size: {n}x{n}, Condition Number: {cond_A:.2e}")
    print(f"Initial Guess Mode: {initial_guess_mode}")
    print(f"{'='*60}")

    results = {}
    condition_data = {"original": cond_A}

    if prrlu_rows is None:
        prrlu_rows = n // 2

    method_runs = [
        (
            "No Precond",
            lambda: newton_schulz_no_precond(
                A,
                max_iters=max_iters,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_unperturbed,
            ),
        ),
        (
            "ILU Precond",
            lambda: newton_schulz_with_ilu_precond(
                A,
                max_iters=max_iters,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_unperturbed,
                initial_guess_seed=initial_guess_seed,
            ),
        ),
        (
            f"prrLU Precond (k={prrlu_rows})",
            lambda: newton_schulz_with_prrlu_precond(
                A,
                max_iters=max_iters,
                num_rows=prrlu_rows,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_unperturbed,
                initial_guess_seed=initial_guess_seed,
            ),
        ),
    ]

    for idx, (label, runner) in enumerate(method_runs, start=1):
        print(f"\n[{idx}] {label}")
        G, errors, iters, info = runner()
        final_error = errors[-1] if errors else float('inf')
        print(f"    Status: {info['status']}")
        if "num_rows" in info:
            print(f"    Rows processed: {info['num_rows']}/{n}")
        if "cond_A_precond" in info:
            print(f"    Precond Condition: {info['cond_A_precond']:.2e} (from {cond_A:.2e})")
        print(f"    Iterations: {iters}, Final Error: {final_error:.6e}")
        results[label] = errors
        if "ILU" in label and "cond_A_precond" in info:
            condition_data["ILU"] = info['cond_A_precond']
        elif "prrLU" in label and "cond_A_precond" in info:
            condition_data["prrLU"] = info['cond_A_precond']

    return {
        "results": results,
        "condition_data": condition_data,
        "matrix_name": matrix_name,
    }


# =============================================================================
# Multiple Trial Testing Functions
# =============================================================================

def run_single_trial(
    matrix_size: int,
    seed: int,
    max_iters: int = DEFAULT_MAX_ITERS,
    prrlu_rows: int | None = None,
    initial_guess_mode: int = 0,
) -> Dict[str, Any]:
    """
    Run a single trial for one matrix configuration.

    Uses:
    - original unperturbed invertible matrix
    - perturbed invertible matrix used for the actual test

    Stores original/perturbed matrices and inverses in `matrix_array`.
    """
    A_original, A, matrix_array, used_seed = _generate_invertible_matrix_with_perturbation(
        n=matrix_size,
        seed=seed,
    )

    n = A.shape[0]
    cond_A = _safe_condition_number(A)
    cond_A_original = _safe_condition_number(A_original)

    if prrlu_rows is None:
        prrlu_rows = n // 2

    trial_results = {
        "cond_A": cond_A,
        "cond_A_original": cond_A_original,
        "seed_used": used_seed,
        "perturbation_scale": DEFAULT_PERTURBATION_SCALE,
        "initial_guess_mode": initial_guess_mode,
        "matrix_array": matrix_array,
    }

    method_runs = [
        (
            "No Precond",
            lambda: newton_schulz_no_precond(
                A,
                max_iters=max_iters,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_original,
            ),
        ),
        (
            "ILU Precond",
            lambda: newton_schulz_with_ilu_precond(
                A,
                max_iters=max_iters,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_original,
                initial_guess_seed=used_seed + 200_001,
            ),
        ),
        (
            f"prrLU Precond (k={prrlu_rows})",
            lambda: newton_schulz_with_prrlu_precond(
                A,
                max_iters=max_iters,
                num_rows=prrlu_rows,
                initial_guess_mode=initial_guess_mode,
                A_unperturbed=A_original,
                initial_guess_seed=used_seed + 300_001,
            ),
        ),
    ]

    for label, runner in method_runs:
        G, errors, iters, info = runner()
        trial_results[label] = {
            "errors": errors,
            "final_error": errors[-1] if errors else float('inf'),
            "iterations": iters,
            "status": info.get("status", "unknown"),
            "initial_guess_norm": info.get("initial_guess_norm", float("nan")),
        }
        if "cond_A_precond" in info:
            trial_results[label]["cond_precond"] = info["cond_A_precond"]
        if "cond_A_unperturbed_precond" in info:
            trial_results[label]["cond_A_unperturbed_precond"] = info["cond_A_unperturbed_precond"]

    return trial_results


def run_multi_trial_test(
    matrix_name: str,
    matrix_size: int = DEFAULT_MATRIX_SIZE,
    n_trials: int = DEFAULT_N_TRIALS,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed_base: int = DEFAULT_SEED_BASE,
    prrlu_rows: int | None = None,
    verbose: bool = True,
    initial_guess_mode: int = 0,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-Trial Test: {matrix_name}")
        print(f"Trials: {n_trials}, Size: {matrix_size}x{matrix_size}")
        print(f"Perturbation scale: {DEFAULT_PERTURBATION_SCALE:.2e}")
        print(f"Initial guess mode: {initial_guess_mode}")
        print(f"{'='*60}")

    trials = []

    for trial_idx in range(n_trials):
        seed = seed_base + trial_idx * 1000
        trial_result = run_single_trial(
            matrix_size=matrix_size,
            seed=seed,
            max_iters=max_iters,
            prrlu_rows=prrlu_rows,
            initial_guess_mode=initial_guess_mode,
        )
        trials.append(trial_result)
        if verbose:
            print(f"  Trial {trial_idx + 1}/{n_trials} completed")

    methods = [k for k in trials[0].keys() if k not in {
        "cond_A", "cond_A_original", "seed_used", "perturbation_scale", "initial_guess_mode", "matrix_array"
    }]

    stats = {}
    avg_errors = {}

    for method in methods:
        final_errors = [t[method]["final_error"] for t in trials]
        iterations = [t[method]["iterations"] for t in trials]
        statuses = [t[method].get("status", "unknown") for t in trials]

        converged_mask = [status == "converged" for status in statuses]
        converged_errors = [e for e, ok in zip(final_errors, converged_mask) if ok and np.isfinite(e)]
        converged_iterations = [it for it, ok in zip(iterations, converged_mask) if ok]

        stats[method] = {
            "final_error_mean": np.mean(converged_errors) if converged_errors else float('inf'),
            "final_error_median": np.median(converged_errors) if converged_errors else float('inf'),
            "final_error_std": np.std(converged_errors) if len(converged_errors) > 1 else 0.0,
            "final_error_min": np.min(converged_errors) if converged_errors else float('inf'),
            "final_error_max": np.max(converged_errors) if converged_errors else float('inf'),
            "iterations_mean": np.mean(converged_iterations) if converged_iterations else float('inf'),
            "iterations_median": np.median(converged_iterations) if converged_iterations else float('inf'),
            "iterations_std": np.std(converged_iterations) if len(converged_iterations) > 1 else 0.0,
            "n_converged": int(sum(converged_mask)),
            "n_trials": n_trials,
            "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        }

        if "cond_precond" in trials[0][method]:
            cond_preconds = [t[method]["cond_precond"] for t in trials]
            stats[method]["cond_precond_mean"] = np.mean(cond_preconds)
            stats[method]["cond_precond_median"] = np.median(cond_preconds)

        max_len = max(len(t[method]["errors"]) for t in trials)
        padded_errors = []
        for t in trials:
            errors = t[method]["errors"]
            if len(errors) < max_len:
                if errors:
                    errors = errors + [errors[-1]] * (max_len - len(errors))
                else:
                    errors = [float('inf')] * max_len
            padded_errors.append(errors)

        padded_errors = np.array(padded_errors, dtype=float)
        avg_errors[method] = np.median(padded_errors, axis=0).tolist()

    cond_As = [t["cond_A"] for t in trials]
    cond_As_original = [t["cond_A_original"] for t in trials]
    stats["cond_A_mean"] = np.mean(cond_As)
    stats["cond_A_median"] = np.median(cond_As)
    stats["cond_A_original_mean"] = np.mean(cond_As_original)
    stats["cond_A_original_median"] = np.median(cond_As_original)

    if verbose:
        print(f"\n  --- Summary Statistics (Converged Runs Only) ---")
        print(f"  Original Unperturbed Condition Number: {stats['cond_A_original_median']:.2e}")
        print(f"  Perturbed Condition Number:           {stats['cond_A_median']:.2e}")
        for method in methods:
            s = stats[method]
            print(f"\n  [{method}]")
            print(f"    Final Error: {s['final_error_median']:.6e} (±{s['final_error_std']:.2e})")
            print(f"    Iterations:  {s['iterations_median']:.1f} (±{s['iterations_std']:.1f})")
            print(f"    Converged:   {s['n_converged']}/{s['n_trials']}")
            print(f"    Statuses:    {s['status_counts']}")
            if "cond_precond_median" in s:
                print(f"    Precond Cond: {s['cond_precond_median']:.2e}")

    return {
        "trials": trials,
        "stats": stats,
        "avg_errors": avg_errors,
        "matrix_name": matrix_name,
    }


def run_comprehensive_multi_trial_test(
    n_trials: int = DEFAULT_N_TRIALS,
    matrix_size: int = DEFAULT_MATRIX_SIZE,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed_base: int = DEFAULT_SEED_BASE,
    test_cases: List[str] | None = None,
    initial_guess_mode: int = 1,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Run multi-trial tests. Matrices are np.random.randn(n,n) with seed.
    Each trial uses an invertible original matrix, adds Gaussian perturbation, and
    tests on the perturbed invertible matrix.
    """
    if test_cases is None:
        test_cases = ["Random"]

    print("\n" + "=" * 70)
    print("PRECONDITIONING TEST")
    print(f"Trials: {n_trials}, Size: {matrix_size}x{matrix_size}")
    print(f"Perturbation scale: {DEFAULT_PERTURBATION_SCALE:.2e}")
    print(f"Initial guess mode: {initial_guess_mode}")
    print("=" * 70)

    all_avg_results = {}
    all_condition_data = {}
    all_detailed_results = {}

    for idx, name in enumerate(test_cases):
        result = run_multi_trial_test(
            matrix_name=name,
            matrix_size=matrix_size,
            n_trials=n_trials,
            max_iters=max_iters,
            seed_base=seed_base + idx * 10000,
            initial_guess_mode=initial_guess_mode,
        )

        all_avg_results[name] = result["avg_errors"]
        all_detailed_results[name] = result

        stats = result["stats"]
        methods = [k for k in result["avg_errors"].keys()]
        all_condition_data[name] = {
            "original_unperturbed": stats["cond_A_original_median"],
            "original": stats["cond_A_median"],
        }
        for method in methods:
            if method in stats and "cond_precond_median" in stats[method]:
                if "ILU" in method:
                    all_condition_data[name]["ILU"] = stats[method]["cond_precond_median"]
                elif "prrLU" in method:
                    all_condition_data[name]["prrLU"] = stats[method]["cond_precond_median"]

    print("\n" + "=" * 70)
    print("PLOTTING: Convergence, Iterations (aggregated)")
    print("=" * 70)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_aggregated_comparison(all_detailed_results, save_dir=out_dir)

    return all_avg_results, all_condition_data, all_detailed_results


if __name__ == "__main__":
    print("=" * 70)
    print("PRECONDITIONING COMPARISON FOR NEWTON-SCHULZ")
    print("=" * 70)

    all_avg_results, all_condition_data, all_detailed_results = run_comprehensive_multi_trial_test(
        n_trials=DEFAULT_N_TRIALS,
        matrix_size=DEFAULT_MATRIX_SIZE,
        max_iters=DEFAULT_MAX_ITERS,
        seed_base=DEFAULT_SEED_BASE,
        initial_guess_mode=1,
    )

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
