import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems.many import DTLZ4
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

# ==============================
# Helper: Spread (Δ) metric
# ==============================
def spread_delta(F, pf):
    # Sort solutions based on first objective
    F = F[np.argsort(F[:, 0])]
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)
    d_mean = np.mean(d)
    df = np.linalg.norm(F[0] - pf[0])
    dl = np.linalg.norm(F[-1] - pf[-1])
    delta = (df + dl + np.sum(np.abs(d - d_mean))) / (df + dl + (len(d) - 1) * d_mean)
    return delta

# Configuration
problem = DTLZ4(n_var=7, n_obj=3)
n_generations = 200
pop_size = 100

partition_methods = ["das-dennis", "energy"]
colors = ["red", "blue"]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot true Pareto front
ref_dirs_default = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = problem.pareto_front(ref_dirs_default)
ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], color="black", s=10, alpha=0.5, label="True Pareto Front")

for method, color in zip(partition_methods, colors):
    if method == "das-dennis":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    elif method == "energy":
        ref_dirs = get_reference_directions("energy", 3, n_points=pop_size)
    else:
        raise ValueError(f"Unsupported method: {method}")

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_generations),
                   seed=1,
                   verbose=False)

    F = res.F
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], color=color, s=20, label=f"NSGA-III ({method})")

    spread = spread_delta(F, pf)
    print(f"Spread (Δ) for {method}: {spread:.4f}")

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_zlabel("Objective 3")
ax.set_title("NSGA-III on DTLZ4 with Partition Methods")
ax.legend()
plt.show()
