from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
import numpy as np

def evaluate_algorithm_on_problem(problem, algorithm, seed=1, verbose=False):
    result = minimize(problem,
                      algorithm,
                      seed=seed,
                      verbose=verbose)

    f_best = result.F
    x_best = result.X

    if problem.n_obj == 1:
        # Single-objective: f_best is a scalar
        print("\nBest solution (X):", x_best)
        print("Objective value (F):", f_best)

        f_opt = getattr(problem, "f_opt", None)
        if callable(f_opt):
            f_opt = f_opt()

        if f_opt is not None:
            f_error = np.abs(f_best - f_opt)
            print("Known optimal (f_opt):", f_opt)
            print("Absolute error from optimum:", f_error)
        else:
            f_error = None
            print("No known optimal value for this problem.")

        return {
            "f_best": f_best,
            "x_best": x_best,
            "f_opt": f_opt,
            "f_error": f_error
        }

    else:
        # Multi-objective: f_best and x_best are arrays of nondominated solutions
        ref_point = np.max(f_best, axis=0) + 0.1
        hv = HV(ref_point=ref_point)
        hv_score = hv.do(f_best)

        print("\nBest solution (1st X):", x_best[0])
        print("Objective value (1st F):", f_best[0])
        print("Hypervolume:", hv_score)

        return {
            "F": f_best,
            "X": x_best,
            "hypervolume": hv_score,
            "ref_point": ref_point
        }

# -------------------
# ✨ Change problem here!
# -------------------
print("hello")

# Try with single-objective:
# problem = get_problem("ackley", n_var=2)

# Try with multi-objective:
problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

results = evaluate_algorithm_on_problem(problem, algorithm, verbose=False)

print("goodbye")