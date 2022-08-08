import numpy as np
import sklearn.gaussian_process.kernels as kernels
from src.bayesian_optimization import bayesian_optimization, BeyesianOptimizationWithCstr
from src.benchmark_functions_2D import *
import json
import argparse

# Set seed
np.random.seed(0)

# Benchmark Function
function_dict = {'bird':Bird(), 'disk':Disk(), 'ackley': Ackley()}

# Communication network
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1
# N = np.ones([1,1])

parser = argparse.ArgumentParser()
parser.add_argument('--objective', type=str, default='ackley')
parser.add_argument('--constraint', type=str, default='disk')
# parser.add_argument('--arg_max', type=np.ndarray, default=None)
parser.add_argument('--n_workers', type=int, default=3)
parser.add_argument('--kernel', type=str, default='RBF')
parser.add_argument('--acquisition_function', type=str, default='ei')
parser.add_argument('--policy', type=str, default='greedy')
parser.add_argument('--fantasies', type=int, default=1)
parser.add_argument('--regularization', type=str, default=None)
parser.add_argument('--regularization_strength', type=float, default=0.01)
parser.add_argument('--pending_regularization', type=str, default=None)
parser.add_argument('--pending_regularization_strength', type=float, default=0.01)
parser.add_argument('--grid_density', type=int, default=30)
parser.add_argument('--n_iters', type=int, default=50)
parser.add_argument('--n_runs', type=int, default=1)
args = parser.parse_args()

assert args.n_workers == N.shape[0]
if function_dict.get(args.objective).arg_min is not None:
  arg_max = function_dict.get(args.objective).arg_min
else:
  arg_max = None

# Bayesian optimization object
BO = BeyesianOptimizationWithCstr(objective = function_dict.get(args.objective),
                                  cstr = function_dict.get(args.constraint),
                                  domain = function_dict.get(args.objective).domain,
                                  arg_max = arg_max,
                                  n_workers = args.n_workers,
                                  network = N,
                                  kernel = kernels.RBF(), # length_scale_bounds=(1, 1000.0) remove this greatly improve performance?
                                  # kernel= kernels.Matern(length_scale=(1, 1000.0)),
                                  acquisition_function = args.acquisition_function,
                                  policy = args.policy,
                                  fantasies = args.fantasies,
                                  regularization = args.regularization,
                                  regularization_strength = args.regularization_strength,
                                  pending_regularization = args.pending_regularization,
                                  pending_regularization_strength = args.pending_regularization_strength,
                                  grid_density = args.grid_density,
                                  args = args)

# Optimize
BO.optimize(n_iters = args.n_iters, n_runs = args.n_runs, n_pre_samples = 5, random_search = 1000, plot = 10)
for a in range(BO.n_workers):
    print("Predicted max {}: {}".format(a, BO.pre_max[a]))
