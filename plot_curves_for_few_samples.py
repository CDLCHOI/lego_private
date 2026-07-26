import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':
    with_mu = False
    # choose = 'FID'; smoothing_weight = 0.7
    choose = 'R-Precision (Top3)'; smoothing_weight = 0.6
    # choose = 'MSE Loss'; smoothing_weight = 0.9
    # choose = 'Skating Ratio'; smoothing_weight = 0.6
    file_list = [
        '/home/deli/project/reward_mdm/output/0911_MDMCLIP_b64_2000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_500/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_1000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_2000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_3000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_4000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_6000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_8000/run.log',
    ]
    
    label_list = [
        'MDM-2000',
        'AdaQF-500',
        'AdaQF-1000',
        'AdaQF-2000',
        'AdaQF-3000',
        'AdaQF-4000',
        'AdaQF-6000',
        'AdaQF-8000',
    ]
    opacity = '60'
    color_list = [
        '#425066'+opacity,
        '#00F954'+opacity,
        '#12B5CB'+opacity,
        '#F90031',
        '#9334E6'+opacity,
        '#F9AB00'+opacity,
        '#7CB342'+opacity,
        '#E52592'+opacity
    ]
    down_sampling_rate = 1 if choose in ['FID', 'R-Precision (Top3)', 'Skating Ratio'] else 1.0

    collect = False

    # smoothing_weight = 0.7 if choose in ['FID', 'R-Precision (Top3)'] else 0.5

    fig, ax = plt.subplots(figsize=(5, 3.8))
    

    def format_func(value, tick_number):
        return f'{int(value/1000)}K'

    i = -1
    for file, label, color in zip(file_list, label_list, color_list):
        i += 1
        # if i==0:
        #     continue
        if i==4:
            break

        f = open(file, 'r')
        lines = f.readlines()
        iters = []
        FIDs = []
        top3s = []
        loss = []
        skatings = []
        epochs = []
        print("file = ", file)

        if 'MDM' in file:
            for l in lines:
                if 'Train. Iter' in l:
                    iter = int(l.split(' ')[6].strip())
                    if iter==200000:
                        break
                    if 'lora' in file:
                        if iter % 2000 == 0:
                            collect = True
                            iters.append(iter)
                    else:
                        if iter % 4000 == 0:
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
                        assert len(FIDs) == len(iters), f'{len(FIDs)}!= {len(iters)}'

                if choose == 'R-Precision (Top3)':
                    if '--> [vald] R_precision:' in l and collect:
                        top3 = float(l.split(' ')[-2].strip())
                        if top3s == []:
                            top3s.append(top3)
                        else:
                            smooth_top3 = top3s[-1] * smoothing_weight + top3 * (1 - smoothing_weight)
                            top3s.append(smooth_top3)
                        collect = False

                # if choose == 'Skating Ratio':
                #     if '---> [vald] Skating Ratio:' in l and collect:
                #         Skating_Ratio = float(l.split(' ')[-1].strip())
                #         if skatings == []:
                #             skatings.append(Skating_Ratio)
                #         else:
                #             smooth_skating_ratio = skatings[-1] * smoothing_weight + Skating_Ratio * (1 - smoothing_weight)
                #             skatings.append(smooth_skating_ratio)
                #         collect = False
                  

        if choose == 'FID':
            if with_mu:
                mean = np.mean(FIDs[80:]); label += fr', $\mu$=({mean:.3f})'
            ax.plot(iters, FIDs, label=label, color=color)
        if choose == 'R-Precision (Top3)':
            if with_mu:
                mean = np.mean(top3s[80:]); label += fr', $\mu$=({mean:.3f})'
            ax.plot(iters, top3s, label=label, color=color)
        # if choose == 'MSE Loss':
        #     if i==0 or i==5:
        #         ax.plot(iters, loss, label=label, color=color)
        # if choose == 'Skating Ratio':
        #     if with_mu:
        #         mean = np.mean(skatings[80:]); label += fr', $\mu$=({mean:.3f})'
        #     ax.plot(iters, skatings, label=label, color=color)
        

    ax.set_xlabel('Iterations')
    # plt.ylabel(choose)
    ax.set_title(choose)
    if choose == 'FID':
        # ax.set_yscale('log')
        ax.set_ylim(0.0, 2.0)
        ax.legend(loc='upper right', fontsize=8)
        # ax.legend(loc='upper right', prop={'size': 8, 'weight': 'bold'})
    if choose == 'R-Precision (Top3)':
        ax.set_ylim(0.65, 0.9)
        ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, zorder=0) # 设置横向虚线
        ax.set_yticks(np.arange(0.60, 0.95, 0.05))
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

