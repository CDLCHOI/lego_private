import matplotlib.pyplot as plt
import numpy as np

# 实验名称
group_labels = ['MDM', 'MoMask', 'SALAD']  # 两组实验
sub_experiments = ['Vanilla', r'+AdaCLIP$\ddag$', r'+AdaCLIP$\S$']  # 子实验

# 示例数值
# 第一行：MoMask系列的三个子实验值
# 第二行：SALAD系列的三个子实验值
values = [
    [0.322, 0.147, 0.177],
    [0.204, 0.136, 0.162],
    [0.296, 0.153, 0.197]
]

# 设置柱状图参数
n_groups = len(group_labels)        # 组数
n_sub = len(sub_experiments)        # 每组子实验数
bar_width = 0.22                    # 每个子实验柱宽
group_gap = 0.5                     # 组间间隔

# 计算每个柱子的位置
indices = np.arange(n_groups)       # 组的位置索引

# 创建图形
plt.figure(figsize=(5, 4))

# 为每组的每个子实验绘制柱子
colors = ['#12B5CB', '#E936B0', '#FED961']  # 同组内不同子实验的颜色
for i in range(n_sub):
    # 计算当前子实验在各组中的位置
    positions = indices * (n_sub * bar_width + group_gap) + i * bar_width
    plt.bar(positions, [values[g][i] for g in range(n_groups)], 
            bar_width, label=sub_experiments[i], color=colors[i], edgecolor=None)

# 添加标签和标题
# plt.ylabel('准确率 (%)', fontsize=12)
plt.title('FID', fontsize=14, pad=20)
plt.xticks(indices * (n_sub * bar_width + group_gap) + (n_sub - 1) * bar_width / 2, 
           group_labels, fontsize=11)
plt.legend(fontsize=9, loc='upper right')

# 在柱子上方添加数值标签
def add_labels():
    for i in range(n_sub):
        positions = indices * (n_sub * bar_width + group_gap) + i * bar_width
        for g in range(n_groups):
            height = values[g][i]
            plt.text(positions[g], height,
                     f'{values[g][i]}', ha='center', va='bottom', fontsize=7)
add_labels()

# 设置y轴范围
plt.ylim(0, 0.5)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
    