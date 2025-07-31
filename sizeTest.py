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

# Configuration
problem = DTLZ4(n_var=7, n_obj=3)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = problem.pareto_front(ref_dirs)
ref_point = np.max(pf, axis=0) * 1.1

n_trials = 30
n_generations = 200

plt.figure(figsize=(8, 6))
cmap = cm.plasma

scatter_points = []
pop_sizes = []

for i in range(n_trials):
    print("a")
    intrestVar = 20 * i + 100
    algorithm = NSGA2(pop_size=intrestVar)

    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=i,
        verbose=False
    )

    elapsed_time = time.time() - start_time

    hv = HV(ref_point=ref_point)
    hv_value = hv(res.F)

    color = cmap(i / n_trials)
    point = plt.scatter(elapsed_time, hv_value, color=color, s=50)
    
    scatter_points.append(point)
    pop_sizes.append(intrestVar)  # Store population size for this point
    print(i)

plt.xlabel("Time (s)")
plt.ylabel("Hypervolume")
plt.title("NSGA-II: Hypervolume vs Time across Multiple Trials")
plt.grid(True)

# Add interactive hover tooltips showing population size
cursor = mplcursors.cursor(scatter_points, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = scatter_points.index(sel.artist)
    sel.annotation.set_text(f"Population Size: {pop_sizes[idx]}")

plt.show()
print("done")
