from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.algorithms.hyperparameters import HyperparameterProblem, MultiRun, stats_single_objective_mean
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


algorithm = NSGA3(ref_dirs=ref_dirs)

problem = Sphere(n_var=10)
n_evals = 500
seeds = [5, 50, 500]

performance = MultiRun(problem, seeds=seeds, func_stats=stats_single_objective_mean, termination=("n_evals", n_evals))

res = minimize(HyperparameterProblem(algorithm, performance),
               MixedVariableGA(pop_size=5),
               termination=('n_evals', 50),
               seed=1,
               verbose=True)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(problem, algorithm, termination=("n_evals", n_evals), seed=6)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))