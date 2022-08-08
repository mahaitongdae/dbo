import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import csv
import __main__
import json


fig, ax = plt.subplots()

result_dirs = ['temp/bird/2022-08-07_212717',
               'temp/bird/2022-08-07_235550',
               'temp/bird/2022-08-09_000753DMCA',
               'temp/bird/2022-08-08_004409',]

# legends = ['distributed', 'regularized', 'expected', 'single_agent']

def _plot_regret(result_dir, log=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
    file_dir = os.path.join(root_dir, result_dir)
    file = os.path.join(file_dir,'data/data.csv')
    param = json.loads(open(os.path.join(file_dir,'data/config.json')).read())
    # identify legend
    if param['n_workers'] > 1:
        if param['fantasies']:
            auto_legend = 'DMCA'
        elif param['regularization'] is not None:
            auto_legend = 'DDR'
        elif param['policy'] == 'safeopt':
            auto_legend = 'StageOpt'
        else:
            auto_legend = 'DCWEI'
    else:
        auto_legend = 'SA'


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
    plt.fill_between(iter, upper, lower, alpha=0.3)
    return use_log_scale, auto_legend

use_log_scale = False
legends = []
real_legends = []
for result_dir in result_dirs:
    log, auto_legend = _plot_regret(result_dir)
    use_log_scale = use_log_scale or log
    legends += [auto_legend, '']
    real_legends.append((auto_legend))

ax.legend(legends)
h = ax.get_legend().legendHandles
handles = []
for i in range(len(real_legends)):
    handles.append(h[2 * i])

ax.get_legend().remove()
ax.legend(handles, real_legends)

plt.xlabel('iterations')
plt.ylabel('immediate regret')
# plt.legend(legends)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
objective = result_dirs[0].split('/')[1]
if use_log_scale:
    plt.savefig(root_dir + '/temp/'+ objective +'/regret_log.pdf', bbox_inches='tight')
    plt.savefig(root_dir + '/temp/'+ objective +'/regret_log.png', bbox_inches='tight')
else:
    plt.savefig(root_dir + '/temp/'+ objective +'/regret.pdf', bbox_inches='tight')
    plt.savefig(root_dir + '/temp/'+ objective +'/regret.png', bbox_inches='tight')