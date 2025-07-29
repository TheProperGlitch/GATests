from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

print("hello")
algorithm = NSGA2()

problem = Sphere(n_var=10)
n_evals = 500

performance = SingleObjectiveSingleRun(problem, termination=("n_evals", n_evals), seed=1)

res = minimize(HyperparameterProblem(algorithm, performance),
               Optuna(),
               termination=('n_evals', 50),
               seed=1,
               verbose=False)

hyperparams = res.X
print(hyperparams)
set_params(algorithm, hierarchical(hyperparams))

res = minimize(Sphere(), algorithm, termination=("n_evals", n_evals), seed=1)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("goodbye")