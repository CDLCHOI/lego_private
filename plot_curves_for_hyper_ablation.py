import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':
    with_mu = True
    # choose = 'FID'; smoothing_weight = 0.8  
    # choose = 'R-Precision (Top3)'; smoothing_weight = 0.7
    # choose = 'MSE Loss'; smoothing_weight = 0.9
    choose = 'Skating Ratio'; smoothing_weight = 0.8
    file_list = [
        '/home/deli/project/reward_mdm/output/0814_MDMCLIP_b128/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl1_tcl1_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl3_tcl1_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl5_tcl1_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl1_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl3_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl5_0716_scratch/run.log',
    ]
    
    label_list = [
        'MDM (baseline)',
        r'$\lambda_M=1\quad, \lambda_T=1$',
        r'$\lambda_M=3\quad, \lambda_T=1$',
        r'$\lambda_M=5\quad, \lambda_T=1$',
        r'$\lambda_M=10\ , \lambda_T=1$',
        r'$\lambda_M=10\ , \lambda_T=2$',
        # r'$\mathbf{\lambda_M=10}\ , \mathbf{\lambda_T=2}$',
        r'$\lambda_M=10\ , \lambda_T=3$',
        r'$\lambda_M=10\ , \lambda_T=5$',
    ]
    opacity = '40'
    color_list = [
        '#425066'+opacity,
        '#00F954'+opacity,
        '#12B5CB'+opacity,
        '#9334E6'+opacity,
        '#F9AB00'+opacity,
        '#F90031',
        '#7CB342'+opacity,
        '#E52592'+opacity
    ]
    down_sampling_rate = 1 if choose in ['FID', 'R-Precision (Top3)', 'Skating Ratio'] else 1.0

    collect = False

    # smoothing_weight = 0.7 if choose in ['FID', 'R-Precision (Top3)'] else 0.5

    fig, ax = plt.subplots()

    def format_func(value, tick_number):
        return f'{int(value/1000)}K'

    i = 0
    for file, label, color in zip(file_list, label_list, color_list):

        f = open(file, 'r')
        lines = f.readlines()
        iters = []
        FIDs = []
        top3s = []
        loss = []
        skatings = []

        for l in lines:
            if 'Train. Iter' in l:
                iter = int(l.split(' ')[6].strip())
                if iter % (5000*down_sampling_rate) == 0:
                    collect = True
                    iters.append(iter)

            if choose == 'FID':
                if '---> [vald] FID:' in l and collect:
                    FID = float(l.split(' ')[-1].strip())
                    if FIDs == []:
                        FIDs.append(FID)
                    else:
                        smooth_FID = FIDs[-1] * smoothing_weight + FID * (1 - smoothing_weight)
                        FIDs.append(smooth_FID)
                    collect = False

            if choose == 'R-Precision (Top3)':
                if '--> [vald] R_precision:' in l and collect:
                    top3 = float(l.split(' ')[-2].strip())
                    if top3s == []:
                        top3s.append(top3)
                    else:
                        smooth_top3 = top3s[-1] * smoothing_weight + top3 * (1 - smoothing_weight)
                        top3s.append(smooth_top3)
                    collect = False

            if choose =='MSE Loss':
                if 'motion_loss.' in l and collect:
                    motion_loss = float(l.split(' ')[9].strip())
                    if loss == []:
                        loss.append(motion_loss)
                    else:
                        smooth_motion_loss = loss[-1] * smoothing_weight + motion_loss * (1 - smoothing_weight)
                        loss.append(smooth_motion_loss)
                    collect = False

            if choose == 'Skating Ratio':
                if '---> [vald] Skating Ratio:' in l and collect:
                    Skating_Ratio = float(l.split(' ')[-1].strip())
                    if skatings == []:
                        skatings.append(Skating_Ratio)
                    else:
                        smooth_skating_ratio = skatings[-1] * smoothing_weight + Skating_Ratio * (1 - smoothing_weight)
                        skatings.append(smooth_skating_ratio)
                    collect = False

        if choose == 'FID':
            if with_mu:
                mean = np.mean(FIDs[80:]); label += fr', $\mu$=({mean:.3f})'
            ax.plot(iters, FIDs, label=label, color=color)
        if choose == 'R-Precision (Top3)':
            if with_mu:
                mean = np.mean(top3s[80:]); label += fr', $\mu$=({mean:.3f})'
            ax.plot(iters, top3s, label=label, color=color)
        if choose == 'MSE Loss':
            if i==0 or i==5:
                ax.plot(iters, loss, label=label, color=color)
        if choose == 'Skating Ratio':
            if with_mu:
                mean = np.mean(skatings[80:]); label += fr', $\mu$=({mean:.3f})'
            ax.plot(iters, skatings, label=label, color=color)
        
        i += 1

    ax.set_xlabel('Iterations')
    # plt.ylabel(choose)
    ax.set_title(choose)
    if choose == 'FID':
        # ax.set_yscale('log')
        ax.set_ylim(0, 0.9)
        ax.legend(loc='upper right', fontsize=8)
        # ax.legend(loc='upper right', prop={'size': 8, 'weight': 'bold'})
    if choose == 'R-Precision (Top3)':
        ax.set_ylim(0.66, 0.95)
        ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, zorder=0)
        ax.set_yticks(np.arange(0.7, 0.95, 0.05))
        ax.legend(loc='lower right', fontsize=8)
    if choose =='MSE Loss':
        # ax.set_ylim(0, 0.5)
        ax.legend()
    if choose == 'Skating Ratio':
        ax.set_ylim(0.05, 0.25)
        ax.legend(loc='upper right', fontsize=8)

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.show()

    print()

