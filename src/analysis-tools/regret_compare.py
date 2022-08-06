import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import csv
import __main__


fig = plt.figure()

result_dirs = ['temp/2022-08-02_220048',
               'temp/2022-08-03_141822_with_ridge',
               'temp/2022-08-04_154753_with_pending',
               'temp/2022-08-04_161628_single',]

legends = ['distributed', 'regularized', 'expected', 'single_agent']

def _plot_regret(result_dir, log=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
    file_dir = os.path.join(root_dir, result_dir)
    file = os.path.join(file_dir,'data/data.csv')
    # file = result_dir + '/data/data.csv'
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        regret = []
        regret_err = []
        dist = []
        iter = []
        for row in reader:
            iter.append(int(row[0]))
            regret.append(max(0, float(row[1])))
            regret_err.append(float(row[2]))
            dist.append(10**(-2)*float(row[3]))

    r_mean = regret
    conf95 = regret_err

    use_log_scale = max(r_mean) / min(r_mean) > 10

    if not use_log_scale:
        # absolut error for linear scale
        lower = [r + err for r, err in zip(r_mean, conf95)]
        upper = [r - err for r, err in zip(r_mean, conf95)]
    else:
        # relative error for log scale
        lower = [10 ** (np.log10(r) + (0.434 * err / r)) for r, err in zip(r_mean, conf95)]
        upper = [10 ** (np.log10(r) - (0.434 * err / r)) for r, err in zip(r_mean, conf95)]



    if use_log_scale:
        plt.yscale('log')

    plt.plot(iter, r_mean, '-', linewidth=1)
    # plt.fill_between(iter, upper, lower, alpha=0.3)
    return use_log_scale

use_log_scale = False
for result_dir in result_dirs:
    log = _plot_regret(result_dir)
    use_log_scale = use_log_scale or log

plt.xlabel('iterations')
plt.ylabel('immediate regret')
plt.legend(legends)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
if use_log_scale:
    plt.savefig(root_dir + '/temp/compare/regret_log.pdf', bbox_inches='tight')
    plt.savefig(root_dir + '/temp/compare/regret_log.png', bbox_inches='tight')
else:
    plt.savefig(root_dir + '/temp/compare/regret.pdf', bbox_inches='tight')
    plt.savefig(root_dir + '/temp/compare/regret.png', bbox_inches='tight')