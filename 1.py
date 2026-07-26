import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免中文乱码，如不需要中文可删除）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 定义横纵坐标数组
x_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 自定义10个y轴数值，shape为(10,)
y_array = np.array([5, 8, 3, 9, 6, 12, 7, 10, 4, 11])

# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars = ax.bar(x_array, y_array, color='#1f77b4', width=0.6)

# 在每个柱子上方标注数值
for bar in bars:
    # 获取柱子的高度（即y值）
    height = bar.get_height()
    # 在柱子顶部居中位置添加文本标注
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x坐标（柱子中心）
        height + 0.2,                       # y坐标（柱子顶部上方0.2，避免重叠）
        f'{height}',                        # 要显示的数值
        ha='center',                        # 水平居中
        va='bottom',                        # 垂直靠下
        fontsize=10,                        # 字体大小
        fontweight='bold'                   # 字体加粗
    )

# 设置图表标题和坐标轴标签
ax.set_title('带数值标注的柱状图', fontsize=14, fontweight='bold')
ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)

# 设置x轴刻度（确保和x_array一致）
ax.set_xticks(x_array)

# 设置y轴范围（留出顶部空间，避免数值标注超出图表）
ax.set_ylim(0, max(y_array) + 2)

# 添加网格线（增强可读性）
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()