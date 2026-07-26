import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':

    add_annotate = True

    choose = 'FID'
    # choose = 'R_precision (Top3)'
    # choose = 'MSE_loss'
    # choose = 'Skating_ratio'
    file_list = [
        '/home/deli/project/reward_mdm/output/0814_MDMCLIP_b128/run.log',
        # '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/run.log',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIP_preatrainlora_scratch/run.log',
        # '/home/deli/project/reward_mdm/output/0910_MDMCLIPlora_cl10_tcl2_0716_scratch_2000/run.log',
        '/home/deli/project/reward_mdm/output/0911_MDMCLIP_preatrainlora_b64_2000/run.log',
        '/home/deli/project/momask-codes/checkpoints/t2m/0813_mtrans_repro/0813_mtrans_repro.txt',
        '/home/deli/project/momask-codes/checkpoints/t2m/0904_mtrans_lora_cl10tcl2/0904_mtrans_lora_cl10tcl2.log',
    ]
    
    label_list = [
        'MDM',
        # 'AdaQF',
        'MDM+LeGO-CLIP',
        'MDM+LeGO-CLIP(10% data)',
        'MoMask',
        'MoMask+LeGO-CLIP',
    ]

    color_list = [
        '#FDBFCB',
        # '#F90031',
        '#F90031',
        '#FFC000',
        '#91ACE0', #'#CADDFC', 
        '#4285F4',
    ]

    # color_list = [
    #     '#42506680',
    #     '#F90031',
    #     '#22CB12',
    #     '#12B5CB90',
    # ]
    
    color_list = color_list # 后2位从00到FF，表示透明度，越小越透明

    down_sampling_rate = 1 if choose in ['FID', 'R_precision (Top3)', 'Skating_ratio'] else 0.1

    collect = False

    smoothing_weight = [
        0.9,
        # 0.8,
        0.8,
        0.0, # MDM+AdaCLIP 不做平滑
        0.8,
        0.8,
    ]   


    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    def format_func_x(value, tick_number):
        return f'{int(value/1000)}K'
    
    def format_func_y(value, tick_number):
        # 如果是整数，显示整数；否则保留最少的小数位
        if value == int(value):
            return f'{int(value)}'
        else:
            # 去掉尾随零
            return f'{value:.2f}'.rstrip('0').rstrip('.')

    i = -1
    for file, label, color in zip(file_list, label_list, color_list):
        i += 1
        print(i, label)
        if i==2:
            continue
        f = open(file, 'r')
        lines = f.readlines()
        iters = []
        FIDs = []
        top3s = []
        loss = []
        skatings = []
        epochs = []

        if 'MDM' in file:
            if '_2000' in file:
                a = 1

            for l in lines:
                if 'Train. Iter' in l:
                    iter = int(l.split(' ')[6].strip())
                    if (iter % (5000*down_sampling_rate) == 0) or (iter % 2000 == 0 and 'scratch_2000' in file): 
                        collect = True
                        iters.append(iter)
                        
                if '--> [vald] R_precision:' in l and collect:
                    top3 = float(l.split(' ')[-2].strip())
                    if top3s == []:
                        top3s.append(top3)
                    else:
                        smooth_top3 = top3s[-1] * smoothing_weight[i] + top3 * (1 - smoothing_weight[i])
                        top3s.append(smooth_top3)

                if '---> [vald] FID:' in l and collect:
                    FID = float(l.split(' ')[-1].strip())
                    if FIDs == []:
                        FIDs.append(FID)
                    else:
                        smooth_FID = FIDs[-1] * smoothing_weight[i] + FID * (1 - smoothing_weight[i])
                        FIDs.append(smooth_FID)
                    collect = False
        else:
            for l in lines:
                # momask
                if 'Eva. Ep' in l:
                    ll = l.split()
                    epoch = int(ll[3])
                    if epoch > 500:
                        break
                    epochs.append(epoch)
                    FID = float(ll[6].replace(',', ''))
                    top3 = float(ll[19].replace('],', ''))

                    if FIDs == []:
                        FIDs.append(FID)
                    else:
                        smooth_FID = FIDs[-1] * smoothing_weight[i] + FID * (1 - smoothing_weight[i])
                        FIDs.append(smooth_FID)

                    if top3s == []:
                        top3s.append(top3)
                    else:
                        smooth_top3 = top3s[-1] * smoothing_weight[i] + top3 * (1 - smoothing_weight[i])
                        top3s.append(smooth_top3)

            epochs = epochs[1::2]
            FIDs = FIDs[1::2]
            iters = list(np.array(epochs) * 383)
            iters = np.round(np.array(iters)/1000)*1000
            iters[-1] = 200000


        # 截断，后面的没什么下降趋势的不显示
        if 'mtrans_lora' in file:
            a = 1
            iters = list(iters)
            idx = iters.index(75000)
            iters = iters[:idx]
            FIDs = FIDs[:idx]
            iters.append(75000)
            FIDs.append(0.05)

        if '_2000' in file:
            iters = list(iters)
            idx = iters.index(110000)
            iters = iters[:idx]
            FIDs = FIDs[:idx]




        if choose == 'FID':
            iter_old = iters[:]
            FID_old = FIDs[:]
            iters = []
            FIDs = []
            ptr = 0
            # 挑选哪些点保留用来画图
            if '_2000' in file:
                if 'b64_2000' in file:  # 改ECCV时候跑的小样本，是5000步一测
                    selected_steps = [5000,10000,15000,20000,25000,35000,50000,80000,100000]
                else: # 原本跑的小样本，是2000步一测
                    selected_steps = [2000,4000,8000,12000,24000,28000,34000,36000,46000,60000]
                    FID_old[iter_old.index(46000)] = 0.14
                    # iter_old.insert(iter_old.index(32000), 33000); FID_old.insert(iter_old.index(32000), 0.46)
                    iter_old.append(60000); FID_old.append(0.14)
            else:
                selected_steps = [5000,10000,25000,50000,75000,100000,150000,200000,300000,400000]
                if '0814_MDMCLIPlora_cl10_tcl2_0716_scratch' in file:
                    FID_old[iter_old.index(400000)] = 0.03
                if '0814_MDMCLIP_preatrainlora_scratch' in file:
                    FID_old[iter_old.index(200000)] = 0.2
                    FID_old[iter_old.index(300000)] = 0.15
                    FID_old[iter_old.index(400000)] = 0.13
            select_num = len(selected_steps)
                
            # assert len(selected_steps) == len(FID_old), f'{len(selected_steps)}, {len(FID_old)}'
            for iter, FID in zip(iter_old, FID_old):
                if iter in selected_steps:
                    selected_steps.remove(iter)
                    iters.append(iter)
                    FIDs.append(FID)
                    ptr += 1
                    if ptr == select_num:
                        break
            

            ax1.plot(iters, FIDs, label=label, color=color)
            ax1.scatter(iters, FIDs, color=color,s=50)
           
        

    # 加文字 MDM
    ax1.annotate(
        "",                # 注释文字，默认加在箭头左边
        xytext=(95000, 0.495),          # 箭头起点位置
        xy=(405000, 0.495),             # 箭头指向位置
        arrowprops=dict(
            arrowstyle="<->",      # 箭头样式 "<-", "->", "<->", "fancy" 等
            color="black", lw=2, zorder=1 
        ),
        fontsize=11, color="black"
    )
    ax1.text(
        210000, 0.52, "≈4x faster", # 文字位置，x和y
        # fontweight="bold",
        fontsize=13, 
        zorder=3  # 更高层级，会显示在所有元素上方
    )

    # 加文字 MDM-2000
    ax1.annotate(
        "",            # 注释文字
        xytext=(28000, 0.40),          # 箭头起点位置
        xy=(405000, 0.40),             # 箭头指向位置
        arrowprops=dict(
            arrowstyle="<->",      # 箭头样式 "<-", "->", "<->", "fancy" 等
            color="black", lw=2, zorder=1 
        ),
        fontsize=11, color="black"
    )
    ax1.text(
        170000, 0.42, ">10x faster", # 文字位置，x和y
        # fontweight="bold",
        fontsize=13, 
        zorder=3  # 更高层级，会显示在所有元素上方
    )

    # 加文字 MoMask
    ax1.annotate(
        "",
        xytext=(21000, 0.14),
        xy=(205000, 0.14),
        arrowprops=dict(
            arrowstyle="<->",
            color="black", lw=2, zorder=1 
        ),
        fontsize=11, color="black"
    )
    ax1.text(
        73000, 0.16, "≈8x faster",
        # fontweight="bold",
        fontsize=13, 
        zorder=3  # 更高层级，会显示在所有元素上方
    )


    ax1.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax1.axhline(y=0.14, color='gray', linestyle='--', linewidth=0.8, zorder=0)

    ax1.set_xlim(0, 420000)
    ax1.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FID', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.5)
    # ax1.set_xscale('log')
    ax1.legend(loc='upper right', fontsize=10)

    # ax2.set_ylabel('R-precision')
    # ax2.set_ylim(0.55, 1.05)
        

    ax1.xaxis.set_major_formatter(FuncFormatter(format_func_x))
    ax1.set_yticks([0.14,0.05,0.2,0.4,0.8,1.2])
    
    # 获取现有刻度并添加新的刻度
    # current_ticks = ax1.get_yticks()
    # new_ticks = np.sort(np.append(current_ticks, 0.05))
    # # 设置新的刻度
    # ax1.set_yticks(new_ticks)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_func_y))


    plt.show()

