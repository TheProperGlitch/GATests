import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# ==============================
# Helpers
# ==============================
def total_spread(F):
    """Average pairwise distance between solutions (diversity)."""
    return np.mean(pdist(F))

def distance_to_global_minimum(F):
    """Average distance to the ideal (0,0,...,0) point."""
    global_min = np.zeros(F.shape[1])
    return np.mean(np.linalg.norm(F - global_min, axis=1))

def inverted_generational_distance(F, PF):
    """Compute IGD (average distance from PF points to closest F point)."""
    D = cdist(PF, F)
    return np.mean(np.min(D, axis=1))

# ==============================
# Configuration
# ==============================
problem = get_problem("dtlz7")
pf = problem.pareto_front()
ref_point = np.max(pf, axis=0) * 1.1

hv_metric = HV(ref_point=ref_point)

n_generations = 200
pop_size = 10
sbx_probs = np.linspace(0.1, 1.0, 100)

hv_values = []
spread_values = []
distance_values = []
igd_values = []


for p in sbx_probs:
    print(f"Running with SBX prob = {p:.2f}")
    
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=p, eta=15)
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_generations),
                   seed=2,
                   verbose=False)

    F = res.F
    hv_values.append(hv_metric(F))
    spread_values.append(total_spread(F))
    distance_values.append(distance_to_global_minimum(F))
    igd_values.append(inverted_generational_distance(F, pf))

# ==============================
# Plot Results
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# HV vs SBX
axes[0, 0].plot(sbx_probs, hv_values, marker='o', color='blue')
axes[0, 0].set_xlabel("SBX Crossover Probability")
axes[0, 0].set_ylabel("Hypervolume")
axes[0, 0].set_title("Hypervolume vs SBX Probability")
axes[0, 0].grid(True)

# Spread vs SBX
axes[0, 1].plot(sbx_probs, spread_values, marker='o', color='green')
axes[0, 1].set_xlabel("SBX Crossover Probability")
axes[0, 1].set_ylabel("Total Spread")
axes[0, 1].set_title("Spread vs SBX Probability")
axes[0, 1].grid(True)

# Distance to Global Min vs SBX
axes[1, 0].plot(sbx_probs, distance_values, marker='o', color='red')
axes[1, 0].set_xlabel("SBX Crossover Probability")
axes[1, 0].set_ylabel("Distance to Global Minimum")
axes[1, 0].set_title("Convergence (Ideal Distance) vs SBX")
axes[1, 0].grid(True)

# IGD vs SBX
axes[1, 1].plot(sbx_probs, igd_values, marker='o', color='purple')
axes[1, 1].set_xlabel("SBX Crossover Probability")
axes[1, 1].set_ylabel("IGD")
axes[1, 1].set_title("IGD vs SBX Probability")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
