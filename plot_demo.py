import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color='blue', linewidth=2)

# 标注峰值点
ax.annotate(
    "peak value",  # 标注文字
    xy=(np.pi/2, 1),  # 箭头指向的坐标
    xytext=(3, 0.8),  # 文字位置
    arrowprops=dict(
        facecolor="red", 
        shrink=0.05, 
        width=2,
        zorder=1  # 箭头层级（较低）
    ),
    fontsize=12,
    zorder=2  # 文字层级（较高，确保在箭头上层）
)

# 添加普通文本（演示层级效果）
ax.text(
    6, -0.5, "sample text", 
    fontsize=12, 
    bbox=dict(facecolor='white', edgecolor='gray'),
    zorder=3  # 更高层级，会显示在所有元素上方
)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.grid(True, zorder=0)  # 网格层级最低

plt.savefig("plot_with_annotation.pdf")  # 保存为矢量图
plt.show()
