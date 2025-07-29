import json
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.parameters import get_params, flatten, set_params, hierarchical

algorithm = NSGA2()
print(flatten(get_params(algorithm)))