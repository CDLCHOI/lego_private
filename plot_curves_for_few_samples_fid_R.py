import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == '__main__':
    with_mu = False
    choose = 'FID'
    # choose = 'R-Precision (Top3)'
    
    opacity = 'F0'
    fid_color_list = [
        '#425066'+opacity,
        '#F90031'+opacity,
        '#00F954'+opacity,
        '#12B5CB'+opacity,
        '#9334E6'+opacity,
        '#F9AB00'+opacity,
        '#7CB342'+opacity,
        '#E52592'+opacity
    ]

    opacity = '60'
    r_color_list = [
        '#425066'+opacity,
        '#F90031'+opacity,
        '#00F954'+opacity,
        '#12B5CB'+opacity,
        '#9334E6'+opacity,
        '#F9AB00'+opacity,
        '#7CB342'+opacity,
        '#E52592'+opacity
    ]


    fig, ax1 = plt.subplots(figsize=(5, 3.8))
    # ax1 = ax1.twinx()
    
    mdm_num = [500, 1000, 2000, 4000, 6000, 8000]
    mdm_fid = [0.537, 0.143, 0.164, 0.326, 0.375, 0.576]
    mdm_R = [0.641, 0.681, 0.696, 0.712, 0.710, 0.767]

    ours_num = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    ours_fid = [0.320, 0.278, 0.148, 0.100, 0.088, 0.051, 0.038 ]
    ours_R = [0.809, 0.842, 0.866, 0.859, 0.887, 0.889, 0.891]

    salad_num = [500, 1000, 2000, 4000, 6000, 8000]
    salad_fid = [0.896, 0.511, 0.103, 0.049, 0.055, 0.061]
    salad_R = [0.712, 0.754, 0.804, 0.814, 0.83, 0.840, ]

    momask_num = [1000, 2000, 4000, 8000]
    momask_fid = [0.991, 0.452, 0.163, 0.059]
    momask_R = [0.616, 0.664, 0.716, 0.783]

    if choose == 'FID':
        ax1.plot(mdm_num, mdm_fid, label='MDM', color=fid_color_list[0], linewidth=2)
        ax1.plot(momask_num, momask_fid, label='MoMask', color=fid_color_list[3], linewidth=2)
        ax1.plot(salad_num, salad_fid, label='Salad', color=fid_color_list[2], linewidth=2)
        ax1.plot(ours_num, ours_fid, label='Ours', color=fid_color_list[1], linewidth=2)
        ax1.scatter(mdm_num, mdm_fid, color=fid_color_list[0],s=50)
        ax1.scatter(ours_num, ours_fid, color=fid_color_list[1],s=50)
        ax1.scatter(salad_num, salad_fid, color=fid_color_list[2],s=50)
        ax1.scatter(momask_num, momask_fid, color=fid_color_list[3],s=50)
        ax1.set_ylim(0, 0.8)
        ax1.set_title('FID')

    elif choose == 'R-Precision (Top3)':

        ax1.plot(mdm_num, mdm_R, label='MDM', color=r_color_list[0], linewidth=2)
        ax1.plot(momask_num, momask_R, label='MoMask', color=r_color_list[3], linewidth=2)
        ax1.plot(salad_num, salad_R, label='Salad', color=r_color_list[2], linewidth=2)
        ax1.plot(ours_num, ours_R, label='Ours', color=r_color_list[1], linewidth=2)
        ax1.scatter(mdm_num, mdm_R, color=fid_color_list[0],s=50, marker='*')
        ax1.scatter(ours_num, ours_R, color=fid_color_list[1],s=50, marker='*')
        ax1.scatter(salad_num, salad_R, color=fid_color_list[2],s=50, marker='*')
        ax1.scatter(momask_num, momask_R, color=fid_color_list[3],s=50, marker='*')
        ax1.set_ylim(0.60, 1.0)
        ax1.set_title('R-Precision (Top3)')

        
    ax1.set_xlabel('Number of Training Samples')



    ax1.legend(loc='upper right', fontsize=8)


    plt.show()

    print()

