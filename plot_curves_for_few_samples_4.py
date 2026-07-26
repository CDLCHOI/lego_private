import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 画2000样本的，mdm momask salad ours

if __name__ == '__main__':
    with_mu = False
    choose = 'FID'; smoothing_weight = 0.7
    # choose = 'R-Precision (Top3)'; smoothing_weight = 0.6
    # choose = 'MSE Loss'; smoothing_weight = 0.9
    # choose = 'Skating Ratio'; smoothing_weight = 0.6
    file_list = [
        '/home/deli/project/reward_mdm/output/0911_MDMCLIP_b64_2000/run.log',
        '/home/deli/project/momask-codes/checkpoints/t2m/0911_mtrans_repro_2000/0911_mtrans_repro_2000.log',
        '/home/deli/project/salad/checkpoints/t2m/0924_denoiser_2000/run.log',
        '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_2000/run.log',
    ]
    
    label_list = [
        'MDM',
        'MoMask',
        'SALAD',
        'Ours',
    ]
    opacity = 'E0'
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

    i = 0
    for file, label, color in zip(file_list, label_list, color_list):
        if i==5:
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
                        
        else:
            if 'salad' in file:
                smoothing_weight = 0.4
            if 'momask' in file:
                smoothing_weight = 0.8

            for l in lines:
                # momask
                if 'Eva. Ep' in l:
                    ll = l.split()

                    if 'salad' in file:
                        epoch = int(ll[6].split(':')[0])
                    else:
                        epoch = int(ll[3])

                    # if epoch > 500:
                    #     break
                        
                    epochs.append(epoch)

                    if 'salad' in file:
                        FID = float(ll[8].replace(',',''))
                        top3 = float(ll[21].replace('),',''))
                    else:
                        FID = float(ll[6].replace(',', ''))
                        top3 = float(ll[19].replace('],', ''))

                    if FIDs == []:
                        FIDs.append(FID)
                    else:
                        smooth_FID = FIDs[-1] * smoothing_weight + FID * (1 - smoothing_weight)
                        FIDs.append(smooth_FID)

                    if top3s == []:
                        top3s.append(top3)
                    else:
                        smooth_top3 = top3s[-1] * smoothing_weight + top3 * (1 - smoothing_weight)
                        top3s.append(smooth_top3)

            if not 'salad' in file:
                epochs = epochs[1::2]
                FIDs = FIDs[1::2]
                top3s = top3s[1::2]
            
            iters = list(np.array(epochs) * 33) # 因为momask和salad的2000样本训练，一个epoch是33次iter
            iters = np.round(np.array(iters)/1000)*1000

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
        
        i += 1

    ax.set_xlabel('Iterations')
    # plt.ylabel(choose)
    # ax.set_title(choose)
    if choose == 'FID':
        # ax.set_yscale('log')
        ax.set_ylim(0.0, 4.0)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('FID with 2000 training samples')
        # ax.legend(loc='upper right', prop={'size': 8, 'weight': 'bold'})
    if choose == 'R-Precision (Top3)':
        ax.set_ylim(0.4, 0.9)
        ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, zorder=0) # 设置横向虚线
        ax.set_yticks(np.arange(0.40, 0.9, 0.1))
        ax.legend(loc='lower right', fontsize=8)
        ax.set_title('R-Precision (top3) with 2000 training samples')
    if choose =='MSE Loss':
        # ax.set_ylim(0, 0.5)
        ax.legend()
    if choose == 'Skating Ratio':
        ax.set_ylim(0.05, 0.25)
        ax.legend(loc='upper right', fontsize=8)

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))


    plt.show()

    print()

