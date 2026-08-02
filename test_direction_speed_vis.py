"""
可视化对抗文本实验中任意模型、任意类别的生成 motion。

用法:
    python test_direction_speed_vis.py --model LeGO --category fwd_bwd
    python test_direction_speed_vis.py --model MDM --category left_right --template_id 0
    python test_direction_speed_vis.py --model MDM+LeGO-CLIP --category walk_run --sample_idx 0 5 10

可用模型: MDM, LeGO, LeGO-0, MDM+LeGO-CLIP
可用类别: left_right, slow_quick, fwd_bwd, cw_ccw, hand_lr, walk_run
"""
import os
import hashlib
import argparse
import numpy as np

from utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

CACHE_DIR = 'output/direction_speed_cache'
FPS = 20

# 模型列表 (与 test_direction_speed.py 保持同步)
MODELS = ['MDM', 'LeGO', 'LeGO-0', 'MDM+LeGO-CLIP']

# 类别配置 (与 test_direction_speed.py CATEGORIES 保持同步)
_CATEGORIES = [
    {
        'name': 'left_right',
        'label_a': 'left', 'label_b': 'right',
        'pairs': [
            ('A person walks toward left', 'A person walks toward right'),
            ('A person walks to the left', 'A person walks to the right'),
            ('A person turns left and keeps walking', 'A person turns right and keeps walking'),
            ('A man steps to his left side', 'A man steps to his right side'),
            ('A person is walking to the left direction', 'A person is walking to the right direction'),
        ],
    },
    {
        'name': 'slow_quick',
        'label_a': 'quickly', 'label_b': 'slowly',
        'pairs': [
            ('A person walks quickly', 'A person walks slowly'),
            ('A person walks forward quickly', 'A person walks forward slowly'),
            ('A person is walking at a fast pace', 'A person is walking at a slow pace'),
            ('A man walks very quickly', 'A man walks very slowly'),
            ('A person quickly moves forward', 'A person slowly moves forward'),
        ],
    },
    {
        'name': 'fwd_bwd',
        'label_a': 'forward', 'label_b': 'backward',
        'pairs': [
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
        ],
    },
    {
        'name': 'cw_ccw',
        'label_a': 'counterclockwise', 'label_b': 'clockwise',
        'pairs': [
            ('A person walks in a circle counterclockwise', 'A person walks in a circle clockwise'),
            ('A person turns counterclockwise', 'A person turns clockwise'),
            ('A person walks counterclockwise', 'A person walks clockwise'),
            ('A man rotates his body counterclockwise', 'A man rotates his body clockwise'),
            ('A person moves along a counter-clockwise path', 'A person moves along a clockwise path'),
        ],
    },
    {
        'name': 'hand_lr',
        'label_a': 'left hand', 'label_b': 'right hand',
        'pairs': [
            ('A person walks forward while raising their left arm',
             'A person walks forward while raising their right arm'),
            ('A person walks forward and raises their left hand up high',
             'A person walks forward and raises their right hand up high'),
            ('A person is walking with his left arm raised',
             'A person is walking with his right arm raised'),
            ('A person walks while holding their left hand up in the air',
             'A person walks while holding their right hand up in the air'),
            ('A man walks forward and lifts his left arm up',
             'A man walks forward and lifts his right arm up'),
        ],
    },
    {
        'name': 'walk_run',
        'label_a': 'run', 'label_b': 'walk',
        'pairs': [
            ('A person runs forward', 'A person walks forward'),
            ('A person is running', 'A person is walking'),
            ('A man runs forward', 'A man walks forward'),
            ('A person runs in a straight line', 'A person walks in a straight line'),
            ('The person runs forward quickly', 'The person walks forward slowly'),
        ],
    },
]


def get_category(name):
    """按名称查找类别配置"""
    for cat in _CATEGORIES:
        if cat['name'] == name:
            return cat
    raise ValueError(f"未知类别: {name}。可用: {[c['name'] for c in _CATEGORIES]}")


def cache_path(model_name, cat_name, tid, side, text):
    """与 test_direction_speed.py 保持一致的缓存路径"""
    tag = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f'{model_name}_{cat_name}_{tid}_{side}_{tag}.npy')


def main():
    cat_names = [c['name'] for c in _CATEGORIES]

    parser = argparse.ArgumentParser(description='可视化对抗文本实验的生成 motion')
    parser.add_argument('--model', type=str, required=True, choices=MODELS,
                        help=f'模型名称 ({", ".join(MODELS)})')
    parser.add_argument('--category', type=str, required=True, choices=cat_names,
                        help=f'实验类别 ({", ".join(cat_names)})')
    parser.add_argument('--template_id', type=int, default=None,
                        help='只可视化指定模板 (0~4)')
    parser.add_argument('--sample_idx', type=int, nargs='+', default=None,
                        help='只可视化指定下标的样本 (如 --sample_idx 0 5 10)')
    parser.add_argument('--max_per_side', type=int, default=None,
                        help='每侧最多渲染多少个样本 (默认全部)')
    args = parser.parse_args()

    cat = get_category(args.category)
    out_dir = f'visualization/{args.model}_{args.category}'
    os.makedirs(out_dir, exist_ok=True)

    templates = cat['pairs']
    label_a = cat['label_a']
    label_b = cat['label_b']

    print(f'模型: {args.model}')
    print(f'类别: {args.category}  (A="{label_a}" vs B="{label_b}")')
    print(f'模板数: {len(templates)}')
    print(f'输出目录: {out_dir}/')
    print('=' * 60)

    total = 0
    for tid, (text_a, text_b) in enumerate(templates):
        if args.template_id is not None and tid != args.template_id:
            continue

        for side, text, label in [('a', text_a, label_a), ('b', text_b, label_b)]:
            path = cache_path(args.model, args.category, tid, side, text)

            if not os.path.exists(path):
                print(f'[skip] 缓存不存在: {path}')
                continue

            joints = np.load(path)  # (20, 196, 22, 3)
            n_samples = joints.shape[0]

            for i in range(n_samples):
                if args.sample_idx is not None and i not in args.sample_idx:
                    continue
                if i==2:
                    break
                save_path = os.path.join(
                    out_dir,
                    f'tid{tid}_{side}_{label}_sample{i:02d}.mp4'
                )
                title = f'{args.model} | tid={tid} | "{label}": {text}'

                plot_3d_motion(
                    save_path,
                    t2m_kinematic_chain,
                    joints[i],
                    title=title,
                    fps=FPS,
                )
                print(f'[done] {save_path}')
                total += 1

                if args.max_per_side is not None and total >= args.max_per_side * 2 * len(templates):
                    break

    print(f'\n全部完成, 共渲染 {total} 个视频 → {out_dir}/')


if __name__ == '__main__':
    main()
