import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':


    # choose = 'FID'
    # choose = 'R-Precision (Top3)'
    # choose = 'MSE Loss'
    choose = 'Skating Ratio'
    file_list = [
        './output/0814_MDMCLIP_b128/run.log',
        './output/0811_MDMCLIPlora_scratch/run.log',
        './output/0811_MDMCLIP_cl10_tcl1_0716_scratch/run.log',
        # './output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ricglobal1/run.log', # 查看加了ric的脚滑比例如何
        './output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/run.log',
    ][::1]
    
    # label_list = [
    #     'w/o both',
    #     'w/o consistency constraint',
    #     'w/o finetuning CLIP',
    #     'Ours (full model)',
    # ][::-1]

    label_list = [
        'MDM (baseline)',
        '+ tuning CLIP',
        '+ consistency constraints',
        'Ours (full model)',
    ][::1]

    color_list = ['#42506650', '#22CB1250', '#12B5CB90', '#F90031']
    # color_list = color_list[::-1] # 后2位从00到FF，表示透明度，越小越透明

    down_sampling_rate = 1 if choose in ['FID', 'R-Precision (Top3)', 'Skating Ratio'] else 0.1

    collect = False

    smoothing_weight = 0.7 if choose in ['FID', 'R-Precision (Top3)', 'Skating Ratio'] else 0.8


    fig, ax = plt.subplots()

    def format_func(value, tick_number):
        return f'{int(value/1000)}K'

    i = 0
    for file, label, color in zip(file_list, label_list, color_list):
        
        i += 1
        print('i = ', i)
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
                    skating_ratio = float(l.split(' ')[-1].strip())
                    if skatings == []:
                        skatings.append(skating_ratio)
                    else:
                        smooth_skating_ratio = skatings[-1] * smoothing_weight + skating_ratio * (1 - smoothing_weight)
                        skatings.append(smooth_skating_ratio)
                    collect = False

        if choose == 'FID':
            ax.plot(iters, FIDs, label=label, color=color)

        if choose == 'R-Precision (Top3)':
            ax.plot(iters, top3s, label=label, color=color)
        if choose == 'MSE Loss':
            if i in [2,3]:
                continue
            ax.plot(iters, loss, label=label, color=color)
        if choose == 'Skating Ratio':
            ax.plot(iters, skatings, label=label, color=color)


    ax.set_xlabel('Training Iteration')
    # plt.ylabel(choose)
    ax.set_title(choose)
    if choose == 'FID':
        ax.set_ylim(0, 0.9)
        ax.legend(loc='upper right', fontsize=9)

        for a in [0.1,0.5,0.6]:
            ax.axhline(y=a, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    if choose == 'R-Precision (Top3)':
        ax.set_ylim(0.65, 0.95)
        ax.set_yticks(np.arange(0.7, 0.95, 0.05))
        ax.legend(loc='lower right', fontsize=10)
    if choose =='MSE Loss':
        ax.set_ylim(0, 0.3)
        ax.legend()
    if choose == 'Skating Ratio':
        ax.set_ylim(0.05, 0.15)
        ax.legend(loc='upper right', fontsize=10)

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.show()

