import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import csv
import __main__
import json




# bird
# result_dirs = ['temp/bird/2022-08-07_212717',
#                'temp/bird/2022-08-07_235550',
#                'temp/bird/2022-08-09_000753DMCA',
#                'temp/bird/2022-08-08_004409',
#                'temp/bird/2022-09-05_235606safeopt',
#                # 'temp/bird/2022-09-19_235729SA-DR',
#                # 'temp/bird/2022-09-20_000555SA-MCA',
#                'temp/bird/2022-09-20_004634SA-safeopt']
# ackley
result_dirs = ['temp/ackley/2022-09-06_105302DCWEI',
               'temp/ackley/2022-09-06_103120DDR',
               'temp/ackley/2022-09-06_103617DMCA',
               'temp/ackley/2022-09-06_104312SA',
               'temp/ackley/2022-09-06_105843safeopt',
               # 'temp/ackley/2022-09-20_004736SA-MCA',
               # 'temp/ackley/2022-09-20_010803SA-DR',
               'temp/ackley/2022-09-20_004634SA-safeopt',
               ]

# rosenbrock
# result_dirs = ['temp/rosenbrock/2022-09-06_111814DCWEI',
#                'temp/rosenbrock/2022-09-10_220027DDR',
#                'temp/rosenbrock/2022-09-06_112232DMCA',
#                'temp/rosenbrock/2022-09-10_230138SA-CWEI',
#                'temp/rosenbrock/2022-09-06_103427safeopt',
#                # 'temp/rosenbrock/2022-09-19_195718SA-MCA',
#                # 'temp/rosenbrock/2022-09-19_201558SA-DR',
#                'temp/rosenbrock/2022-09-19_234954SA-safeopt',
#                # 'temp/rosenbrock/2022-09-12_122105EI',
#                ]


# legends = ['distributed', 'regularized', 'expected', 'single_agent']

def _plot_regret(result_dir, x_axis = 'iter', log=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
    file_dir = os.path.join(root_dir, result_dir)
    file = os.path.join(file_dir,'data/data.csv')
    param = json.loads(open(os.path.join(file_dir,'data/config.json')).read())
    # identify legend
    if param['fantasies']:
        auto_legend = 'MCA'
    elif param['regularization'] is not None:
        auto_legend = 'DR'
    elif param['acquisition_function'] == 'safeopt':
        auto_legend = 'StageOpt'
    else:
        # if param['unconstrained']:
        #     auto_legend = 'EI'
        # else:
        auto_legend = 'CWEI'

    if param['n_workers'] > 1:
        auto_legend = 'MA-' + auto_legend
    else:
        auto_legend = 'SA-' + auto_legend

    # file = result_dir + '/data/data.csv'
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        regret = []
        regret_err = []
        dist = []
        iter = []
        interactions = []
        for row in reader:
            iter.append(int(row[0]))
            interactions.append(int(row[0])) if auto_legend.startswith('SA')  else interactions.append(3 * int(row[0]))
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

    if x_axis == 'iter':
        plt.plot(iter, r_mean, '-', linewidth=1)
        plt.fill_between(iter, upper, lower, alpha=0.3)
        plt.xlabel('iterations')
    elif x_axis == 'interactions':
        plt.plot(interactions, r_mean, '-', linewidth=1)
        plt.fill_between(interactions, upper, lower, alpha=0.3)
        plt.xlabel('data collected')
    elif x_axis == 'dist':
        plt.plot(dist, r_mean, '-', linewidth=1)
        plt.fill_between(dist, upper, lower, alpha=0.3)
        plt.xlabel('dist traveled')

    return use_log_scale, auto_legend, x_axis

for x_axis in ['iter', 'interactions', 'dist']:

    fig, ax = plt.subplots()
    use_log_scale = False
    legends = []
    real_legends = []
    for result_dir in result_dirs:
        log, auto_legend, x_axis = _plot_regret(result_dir, x_axis=x_axis)
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


    plt.ylabel('immediate regret')
    # plt.legend(legends)
    plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
    objective = result_dirs[0].split('/')[1]
    if use_log_scale:
        plt.savefig(root_dir + '/temp/'+ objective +'/regret_log_' + x_axis + '.pdf', bbox_inches='tight')
        plt.savefig(root_dir + '/temp/'+ objective +'/regret_log_' + x_axis + '.png', bbox_inches='tight')
    else:
        plt.savefig(root_dir + '/temp/'+ objective +'/regret_' + x_axis + '.pdf', bbox_inches='tight')
        plt.savefig(root_dir + '/temp/'+ objective +'/regret_' + x_axis + '.png', bbox_inches='tight')