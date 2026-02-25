import GP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import io, contextlib

np.random.seed(0)

n = 13
sig = 0.01
x = np.random.rand(n)
y = np.sin(x * 2 * np.pi) + sig * np.random.randn(n)

xth = np.linspace(0, 1, 100)
yth = np.sin(xth * 2 * np.pi)

theta0 = np.array([0.1, 0.01])
alpha = 1e-6
max_iter = 80

# --- run with np.linalg.inv ---
gp_inv = GP.GPR(n, 1, "RBF", sig, use_ns=False)
gp_inv.fit(np.array([x]), y, np.array([sig * 10, 1e-2]))

t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    theta_hist_inv = gp_inv.grad_dec_theta(theta0, alpha=alpha, max_iter=max_iter, return_theta=True)
time_inv = time.perf_counter() - t0

pred_inv = gp_inv.predict_aft_K(xth)

# --- run with Newton-Schulz ---
gp_ns = GP.GPR(n, 1, "RBF", sig, use_ns=True)
gp_ns.fit(np.array([x]), y, np.array([sig * 10, 1e-2]))

t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    theta_hist_ns = gp_ns.grad_dec_theta(theta0, alpha=alpha, max_iter=max_iter, return_theta=True)
time_ns = time.perf_counter() - t0

with contextlib.redirect_stdout(io.StringIO()):
    pred_ns = gp_ns.predict_aft_K(xth)

# --- log-likelihood traces (use exact inv for fair LL comparison) ---
ll_inv = []
ll_ns = []
for th in theta_hist_inv:
    gp_inv.set_theta(th)
    gp_inv.ker.compute(gp_inv.X)
    gp_inv.use_ns = False
    ll_inv.append(float(np.real(gp_inv.log_like())))

for th in theta_hist_ns:
    gp_ns.set_theta(th)
    gp_ns.ker.compute(gp_ns.X)
    gp_ns.use_ns = False
    ll_ns.append(float(np.real(gp_ns.log_like())))

print("\n" + "=" * 50)
print(f"np.linalg.inv  time: {time_inv:.4f}s  ({len(theta_hist_inv)-1} iters)")
print(f"Newton-Schulz   time: {time_ns:.4f}s  ({len(theta_hist_ns)-1} iters)")
print(f"Final theta (inv): {theta_hist_inv[-1]}")
print(f"Final theta (NS):  {theta_hist_ns[-1]}")
print("=" * 50)

# --- plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# top-left: GP prediction with np.linalg.inv
ax = axes[0, 0]
ax.scatter(x, y, c='black', s=20, zorder=5, label="true data")
ax.plot(xth, yth, 'k--', alpha=0.5, label="sin(2πx)")
ax.plot(xth, pred_inv, 'b-', linewidth=2, label="predicted (linalg.inv)")
ax.set_title("GP with np.linalg.inv")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# top-right: GP prediction with Newton-Schulz
ax = axes[0, 1]
ax.scatter(x, y, c='black', s=20, zorder=5, label="true data")
ax.plot(xth, yth, 'k--', alpha=0.5, label="sin(2πx)")
ax.plot(xth, pred_ns, 'r-', linewidth=2, label="predicted (Newton-Schulz)")
ax.set_title("GP with Newton-Schulz")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# bottom-left: log-likelihood convergence
ax = axes[1, 0]
ax.plot(ll_inv, 'b.-', label=f"linalg.inv ({time_inv:.3f}s)")
ax.plot(ll_ns, 'r.--', label=f"Newton-Schulz ({time_ns:.3f}s)")
ax.set_title("Log-Likelihood Convergence")
ax.set_xlabel("GD iteration")
ax.set_ylabel("log-likelihood")
ax.legend()

# bottom-right: theta trajectory
ax = axes[1, 1]
th_inv = np.array(theta_hist_inv)
th_ns = np.array(theta_hist_ns)
ax.plot(th_inv[:, 0], th_inv[:, 1], 'b.-', markersize=4, label="linalg.inv")
ax.plot(th_ns[:, 0], th_ns[:, 1], 'r.--', markersize=4, label="Newton-Schulz")
ax.set_title("Hyperparameter Trajectory")
ax.set_xlabel("σ_f")
ax.set_ylabel("ℓ")
ax.legend()

plt.tight_layout()
plt.savefig("ns_vs_inv_comparison.png", dpi=150)
print("Saved ns_vs_inv_comparison.png")
