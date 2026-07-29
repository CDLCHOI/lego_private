"""测试 SnapMoGen 可视化函数。

读取 dataset/SnapMoGen/renamed_feats/dd_00000.npy 的 motion 数据，
调用 visualize_snapmogen 函数将其渲染为 MP4 视频文件。
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.visualize.vis_snapmogen import visualize_snapmogen


def main():
    # 测试数据路径
    motion_path = 'dataset/SnapMoGen/renamed_feats/dd_00000.npy'
    output_dir = 'output/test_vis'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dd_00000.mp4')

    print(f'加载 motion: {motion_path}')
    motion = np.load(motion_path)
    print(f'  motion shape: {motion.shape}')
    print(f'  motion dtype: {motion.dtype}')
    print(f'  motion range: [{motion.min():.4f}, {motion.max():.4f}]')

    # 直接调用可视化（不传 mean/std，因为原始 npy 文件已在原始特征空间）
    print(f'开始渲染...')
    result = visualize_snapmogen(
        motion=motion,
        output_path=output_path,
        caption='dd_00000 (test)',
        mean=None,   # 原始特征空间，不需要反归一化
        std=None,
        fps=30,
    )

    print(f'渲染完成: {result}')
    assert os.path.exists(result), f'MP4 文件未生成: {result}'
    file_size_mb = os.path.getsize(result) / (1024 * 1024)
    print(f'文件大小: {file_size_mb:.2f} MB')
    print('测试通过!')


if __name__ == '__main__':
    main()
