"""
可视化 LeGO 模型在 fwd_bwd (向前/向后走, 面朝前) 类别下的所有生成 motion。

用法:
    python test_direction_speed_vis.py
    python test_direction_speed_vis.py --sample_idx 0 5 10  # 只可视化指定下标的样本
    python test_direction_speed_vis.py --template_id 0        # 只可视化指定模板
"""
import os
import sys
import hashlib
import numpy as np
from utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

CACHE_DIR = 'output/direction_speed_cache'
OUT_DIR = 'visualization/fwd_bwd_vis'
MODEL = 'LeGO'

# fwd_bwd 的5组文本对 (必须与 test_direction_speed.py CATEGORIES[2] 保持一致)
TEMPLATES = [
    ('A person faces forward and walks forward',
     'A person faces forward and walks backward'),
    ('A person is facing forward and walks forward',
     'A person is facing forward and walks backward'),
    ('A person walks forward while facing forward',
     'A person walks backward while facing forward'),
    ('A person is facing forward and steps forward',
     'A person is facing forward and steps backward'),
    ('A person faces forward and walks forward in a straight line',
     'A person faces forward and walks backward in a straight line'),
]
LABELS = ['forward', 'backward']
FPS = 20


def cache_path(model_name, tid, side, text):
    """与 test_direction_speed.py 保持一致的缓存路径"""
    tag = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f'{model_name}_fwd_bwd_{tid}_{side}_{tag}.npy')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 解析命令行过滤参数
    filter_tids = None
    filter_idxs = None
    argv = sys.argv[1:]
    while argv:
        a = argv.pop(0)
        if a == '--template_id' and argv:
            filter_tids = [int(argv.pop(0))]
        elif a == '--sample_idx' and argv:
            filter_idxs = []
            while argv and argv[0].lstrip('-').isdigit():
                filter_idxs.append(int(argv.pop(0)))

    for tid, (text_forward, text_backward) in enumerate(TEMPLATES):
        if filter_tids is not None and tid not in filter_tids:
            continue

        for side, text in [('a', text_forward), ('b', text_backward)]:
            path = cache_path(MODEL, tid, side, text)
            label = LABELS[0 if side == 'a' else 1]

            if not os.path.exists(path):
                print(f'[skip] 缓存不存在: {path}')
                continue

            joints = np.load(path)  # (20, 196, 22, 3)
            n_samples = joints.shape[0]

            for i in range(n_samples):
                if i==2:
                    break
                if filter_idxs is not None and i not in filter_idxs:
                    continue

                save_path = os.path.join(
                    OUT_DIR,
                    f'{MODEL}_tid{tid}_{label}_sample{i:02d}.mp4'
                )
                title = f'tid={tid} "{label}": {text[:60]}...'

                plot_3d_motion(
                    save_path,
                    t2m_kinematic_chain,
                    joints[i],
                    title=title,
                    fps=FPS,
                )
                print(f'[done] {save_path}')

    print(f'\n全部完成, 输出目录: {OUT_DIR}/')


if __name__ == '__main__':
    main()
