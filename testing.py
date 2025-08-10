from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

print("Hello")

# ==============================
# Helper function: Spread (Δ) metric
# ==============================
def spread_delta(F, pf):
    # Sort solutions based on the first objective
    F = F[np.argsort(F[:, 0])]
    
    # Euclidean distances between consecutive solutions
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)
    d_mean = np.mean(d)
    
    # Distances to extremes of Pareto front
    df = np.linalg.norm(F[0] - pf[0])
    dl = np.linalg.norm(F[-1] - pf[-1])
    
    # Calculate spread metric (lower is better)
    delta = (df + dl + np.sum(np.abs(d - d_mean))) / (df + dl + (len(d) - 1) * d_mean)
    return delta

# define problem
problem = get_problem("dtlz7")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

# Calculate spread
pf = problem.pareto_front()
spread = spread_delta(res.F, pf)
print(f"Spread (Δ): {spread:.4f}")

# plotting
plot = Scatter()
plot.add(pf, plot_type="scatter", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

print("goodbye")
