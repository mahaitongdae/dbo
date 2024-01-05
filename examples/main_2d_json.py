import numpy as np
import sklearn.gaussian_process.kernels as kernels
import sys

sys.path.append('/home/mht/PycharmProjects/dbo')

from src.bayesian_optimization import bayesian_optimization, BayesianOptimizationCentralized
from src.benchmark_functions_2D import *
import json
import argparse

# Set seed
np.random.seed(0)

# Benchmark Function
function_dict = {'bird':Bird(), 'disk':Disk(), 'ackley': Ackley(), 'rosenbrock': Rosenbrock(),
                 'eggholder': Eggholder()}
kernel_dict = {'RBF':kernels.RBF(), 'Matern':kernels.Matern()}

# Communication network

# N = np.ones([1,1])

parser = argparse.ArgumentParser()
parser.add_argument('--objective', type=str, default='rosenbrock')
parser.add_argument('--constraint', type=str, default='disk')
parser.add_argument('--model', type=str, default='torch') #torch or sklearn
# parser.add_argument('--arg_max', type=np.ndarray, default=None)
parser.add_argument('--n_workers', type=int, default=30)
parser.add_argument('--kernel', type=str, default='Matern')
parser.add_argument('--acquisition_function', type=str, default='es')
parser.add_argument('--policy', type=str, default='greedy')
parser.add_argument('--unconstrained', type=bool, default=True)
parser.add_argument('--decision_type', type=str, default='parallel')
parser.add_argument('--fantasies', type=int, default=0)
parser.add_argument('--regularization', type=str, default=None)
parser.add_argument('--regularization_strength', type=float, default=0.01)
parser.add_argument('--pending_regularization', type=str, default=None)
parser.add_argument('--pending_regularization_strength', type=float, default=0.01)
parser.add_argument('--grid_density', type=int, default=30)
parser.add_argument('--n_iters', type=int, default=150)
parser.add_argument('--sim', type=bool, default=False)
parser.add_argument('--n_runs', type=int, default=5)
parser.add_argument('--random_search', type=int, default=1000)
parser.add_argument('--diversity_penalty', type=bool, default=False)
parser.add_argument('--div_radius', type=float, default=0.2)
parser.add_argument('--truncated_at_regret', default=None)
args = parser.parse_args()
if args.n_workers == 3:
    N = np.ones([3,3])
    # N[0, 1] = N[1, 0] = N[1, 2] = N[2, 1] = 1
    # args.n_iters = 50
elif args.n_workers == 1:
    N = np.ones([1, 1])
    args.n_iters = 150
else:
    N = np.ones([args.n_workers,args.n_workers])
# assert args.n_workers == N.shape[0]
if function_dict.get(args.objective).arg_min is not None:
  arg_max = function_dict.get(args.objective).arg_min
else:
  arg_max = None

if args.sim:
    args.projection_in_graident_step = True
    args.truncated_at_regret = 1e-1

# Bayesian optimization object
if args.decision_type == 'distributed':
    BO = bayesian_optimization(objective = function_dict.get(args.objective),
                                  domain = function_dict.get(args.objective).domain,
                                  arg_max = arg_max,
                                  n_workers = args.n_workers,
                                  network = N,
                                  kernel = kernel_dict.get(args.kernel), # length_scale_bounds=(1, 1000.0) remove this greatly improve performance?
                                  acquisition_function = args.acquisition_function,
                                  policy = args.policy,
                                  fantasies = args.fantasies,
                                  regularization = args.regularization,
                                  regularization_strength = args.regularization_strength,
                                  pending_regularization = args.pending_regularization,
                                  pending_regularization_strength = args.pending_regularization_strength,
                                  grid_density = args.grid_density,
                                  args = args)
else:
    BO = BayesianOptimizationCentralized(objective=function_dict.get(args.objective),
                               domain=function_dict.get(args.objective).domain,
                               arg_max=arg_max,
                               n_workers=args.n_workers,
                               network=N,
                               kernel=kernel_dict.get(args.kernel),
                               # length_scale_bounds=(1, 1000.0) remove this greatly improve performance?
                               acquisition_function=args.acquisition_function,
                               policy=args.policy,
                               fantasies=args.fantasies,
                               regularization=args.regularization,
                               regularization_strength=args.regularization_strength,
                               pending_regularization=args.pending_regularization,
                               pending_regularization_strength=args.pending_regularization_strength,
                               grid_density=args.grid_density,
                               args=args)

# Optimize
BO.optimize(n_iters = args.n_iters, n_runs = args.n_runs, n_pre_samples = max(15, args.n_workers), random_search = args.random_search, plot = [0, 10, 50, 150])
# for a in range(BO.n_workers):
#     print("Predicted max {}: {}".format(a, BO.pre_max[a]))
