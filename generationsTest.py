import time
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many import DTLZ4
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import matplotlib.cm as cm

# ==============================
# Helper: Spread (Δ) metric
# ==============================
def spread_delta(F, pf):
    """
    Calculates Deb's Spread (Δ) metric for diversity.
    - F : final solutions (objective space)
    - pf : true Pareto front
    """
    # Sort solutions based on the first objective for consistent ordering
    F = F[np.argsort(F[:, 0])]

    # Compute Euclidean distances between consecutive solutions
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)
    d_mean = np.mean(d)

    # Distances to the extremes of the Pareto front
    df = np.linalg.norm(F[0] - pf[0])
    dl = np.linalg.norm(F[-1] - pf[-1])

    # Spread metric formula (lower is better)
    delta = (df + dl + np.sum(np.abs(d - d_mean))) / (df + dl + (len(d) - 1) * d_mean)
    return delta

# ==============================
# Configuration
# ==============================
problem = DTLZ4(n_var=7, n_obj=3)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = problem.pareto_front(ref_dirs)  # True Pareto front for comparison
ref_point = np.max(pf, axis=0) * 1.1

n_trials = 100
fixed_pop_size = 100  # keep population size fixed

plt.figure(figsize=(14, 6))

# Create two subplots: HV and Spread
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

cmap = cm.plasma
scatter_hv = []
scatter_spread = []
generation_counts = []

hv_metric = HV(ref_point=ref_point)

# ==============================
# Run trials with varying generations
# ==============================
for i in range(n_trials):
    n_generations_var = 10 * i + 50  # vary generations per trial
    algorithm = NSGA2(pop_size=fixed_pop_size)
    print(i)
    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations_var),
        seed=i,
        verbose=False
    )
    elapsed_time = time.time() - start_time

    # Compute Hypervolume
    hv_value = hv_metric(res.F)

    # Compute Spread metric
    spread_value = spread_delta(res.F, pf)

    # Plot points with color progression
    color = cmap(i / n_trials)
    p1 = ax1.scatter(elapsed_time, hv_value, color=color, s=50)
    p2 = ax2.scatter(elapsed_time, spread_value, color=color, s=50)

    scatter_hv.append(p1)
    scatter_spread.append(p2)
    generation_counts.append(n_generations_var)

# ==============================
# Configure Plots
# ==============================
ax1.set_title("Accuracy (Hypervolume) vs Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Hypervolume")
ax1.grid(True)

ax2.set_title("Diversity (Spread Δ) vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Spread Δ (Lower is Better)")
ax2.grid(True)

# ==============================
# Interactive tooltips showing n_generations
# ==============================
cursor1 = mplcursors.cursor(scatter_hv, hover=True)
cursor2 = mplcursors.cursor(scatter_spread, hover=True)

@cursor1.connect("add")
def on_add_hv(sel):
    idx = scatter_hv.index(sel.artist)
    sel.annotation.set_text(f"Generations: {generation_counts[idx]}")

@cursor2.connect("add")
def on_add_spread(sel):
    idx = scatter_spread.index(sel.artist)
    sel.annotation.set_text(f"Generations: {generation_counts[idx]}")

plt.tight_layout()
plt.show()
print("done")
