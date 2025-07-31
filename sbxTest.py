import time
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many import DTLZ4
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import matplotlib.cm as cm

# ==============================
# Helper: Spread (Δ) metric
# ==============================
def spread_delta(F, pf):
    F = F[np.argsort(F[:, 0])]
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)
    d_mean = np.mean(d)
    df = np.linalg.norm(F[0] - pf[0])
    dl = np.linalg.norm(F[-1] - pf[-1])
    return (df + dl + np.sum(np.abs(d - d_mean))) / (df + dl + (len(d) - 1) * d_mean)

# ==============================
# Configuration
# ==============================
problem = DTLZ4(n_var=7, n_obj=3)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = problem.pareto_front(ref_dirs)
ref_point = np.max(pf, axis=0) * 1.1

n_trials = 100
n_generations = 200
fixed_pop_size = 100

# Define crossover probabilities to test
crossover_probs = np.linspace(0.5, 1.0, n_trials)  # from 0.5 to 1.0

plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

cmap = cm.plasma
scatter_hv = []
scatter_spread = []
crossover_labels = []

hv_metric = HV(ref_point=ref_point)

# ==============================
# Run trials with varying crossover probability
# ==============================
for i, cx_prob in enumerate(crossover_probs): 
    print(i)
    crossover = SBX(prob=cx_prob, eta=15)  # eta fixed, vary probability
    algorithm = NSGA2(pop_size=fixed_pop_size, crossover=crossover)

    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=i,
        verbose=False
    )
    elapsed_time = time.time() - start_time

    hv_value = hv_metric(res.F)
    spread_value = spread_delta(res.F, pf)

    color = cmap(i / n_trials)
    p1 = ax1.scatter(elapsed_time, hv_value, color=color, s=50)
    p2 = ax2.scatter(elapsed_time, spread_value, color=color, s=50)

    scatter_hv.append(p1)
    scatter_spread.append(p2)
    crossover_labels.append(cx_prob)

# ==============================
# Configure Plots
# ==============================
ax1.set_title("Accuracy (Hypervolume) vs Time (Varying Crossover Prob.)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Hypervolume")
ax1.grid(True)

ax2.set_title("Diversity (Spread Δ) vs Time (Varying Crossover Prob.)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Spread Δ (Lower is Better)")
ax2.grid(True)

# ==============================
# Interactive tooltips showing crossover probability
# ==============================
cursor1 = mplcursors.cursor(scatter_hv, hover=True)
cursor2 = mplcursors.cursor(scatter_spread, hover=True)

@cursor1.connect("add")
def on_add_hv(sel):
    idx = scatter_hv.index(sel.artist)
    sel.annotation.set_text(f"Crossover Prob: {crossover_labels[idx]:.2f}")

@cursor2.connect("add")
def on_add_spread(sel):
    idx = scatter_spread.index(sel.artist)
    sel.annotation.set_text(f"Crossover Prob: {crossover_labels[idx]:.2f}")

plt.tight_layout()
plt.show()
print("done")
