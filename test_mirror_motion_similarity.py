"""
测试原始 motion 和镜像 motion 之间的 embedding 余弦相似度。

读取 dataset/HumanML3D/new_joint_vecs 中的所有原始数据（NNNNNN.npy）和
对应的镜像数据（MNNNNNN.npy），使用 EvaluatorMDMWrapper 提取每个 motion
的 embedding，计算每对原始-镜像 embedding 之间的余弦相似度。

用法:
    python test_mirror_motion_similarity.py
    python test_mirror_motion_similarity.py --verbose   # 打印每对相似度
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def build_dummy_args():
    """创建 EvaluatorMDMWrapper 所需的虚拟 args 命名空间"""
    class DummyArgs:
        pass
    return DummyArgs()


def load_motion(filepath: str) -> np.ndarray:
    """加载单个 .npy motion 文件，返回 (T, 263) 的 float32 数组"""
    data = np.load(filepath)
    if data.ndim != 2:
        raise ValueError(f"期望 2D 数组，但 {filepath} 的 shape 是 {data.shape}")
    return data


def compute_similarity(evaluator, orig_path: str, mirr_path: str):
    """
    计算一对原始-镜像 motion 的 embedding 余弦相似度。

    Returns:
        (orig_len, mirr_len, cosine_similarity)
    """
    orig_data = load_motion(orig_path)       # (T_o, 263)
    mirr_data = load_motion(mirr_path)       # (T_m, 263)

    orig_len = orig_data.shape[0]
    mirr_len = mirr_data.shape[0]

    # 添加 batch 维度 → (1, T, 263)
    orig_tensor = torch.from_numpy(orig_data).float().unsqueeze(0)
    mirr_tensor = torch.from_numpy(mirr_data).float().unsqueeze(0)

    orig_len_t = torch.tensor([orig_len])
    mirr_len_t = torch.tensor([mirr_len])

    # 提取 embedding
    orig_emb = evaluator.get_motion_embeddings(orig_tensor, orig_len_t)   # (1, 512)
    mirr_emb = evaluator.get_motion_embeddings(mirr_tensor, mirr_len_t)   # (1, 512)

    # 余弦相似度
    cos_sim = F.cosine_similarity(orig_emb, mirr_emb, dim=1).item()

    return orig_len, mirr_len, cos_sim


def print_statistics(results: list):
    """打印详细的统计信息"""
    sims = np.array([r['cos_sim'] for r in results])

    print('\n' + '=' * 65)
    print('                    统 计 结 果')
    print('=' * 65)
    print(f'  样本对数:         {len(sims):>8d}')
    print(f'  平均余弦相似度:    {sims.mean():>10.6f}')
    print(f'  标准差:           {sims.std():>10.6f}')
    print(f'  最小值:           {sims.min():>10.6f}  '
          f'(文件: {results[int(np.argmin(sims))]["filename"]})')
    print(f'  最大值:           {sims.max():>10.6f}  '
          f'(文件: {results[int(np.argmax(sims))]["filename"]})')
    print(f'  中位数:           {np.median(sims):>10.6f}')
    print('-' * 65)

    # 分布统计
    thresholds = [0.999, 0.99, 0.95, 0.90, 0.85, 0.80, 0.70]
    print('  相似度分布:')
    for t in thresholds:
        count = int((sims >= t).sum())
        pct = count / len(sims) * 100
        print(f'    >= {t:.3f}:  {count:>6d}  ({pct:>6.2f}%)')

    # 低相似度样本
    low_threshold = 0.80
    low_indices = np.where(sims < low_threshold)[0]
    if len(low_indices) > 0:
        print(f'\n  相似度 < {low_threshold:.2f} 的样本 (共 {len(low_indices)} 个):')
        for idx in low_indices[:20]:  # 最多显示20个
            r = results[idx]
            print(f'    {r["filename"]}: cos_sim={r["cos_sim"]:.6f}, '
                  f'orig_len={r["orig_len"]}, mirr_len={r["mirr_len"]}')
        if len(low_indices) > 20:
            print(f'    ... 还有 {len(low_indices) - 20} 个样本')

    print('=' * 65)


def main():
    parser = argparse.ArgumentParser(
        description='测试原始和镜像 motion 的 embedding 余弦相似度'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='dataset/HumanML3D/new_joint_vecs',
        help='数据目录路径'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        # default='/home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar',
        default='checkpoints/t2m/text_mot_match/model/finest.tar',
        help='Evaluator checkpoint 路径'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='打印每一对数据的余弦相似度'
    )
    parser.add_argument(
        '--print_interval', type=int, default=500,
        help='每 N 个样本打印一次进度摘要（仅在非 verbose 模式）'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='最多处理的样本对数（用于快速测试，默认处理全部）'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='将详细结果保存到指定文件'
    )

    args_cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'设备:              {device}')
    print(f'数据目录:          {args_cli.data_dir}')
    print(f'Checkpoint:        {args_cli.checkpoint}')
    print(f'Verbose 模式:      {args_cli.verbose}')

    # 检查数据目录
    if not os.path.isdir(args_cli.data_dir):
        print(f'\n错误: 数据目录不存在: {args_cli.data_dir}')
        sys.exit(1)

    # 检查 checkpoint
    if not os.path.isfile(args_cli.checkpoint):
        print(f'\n错误: Checkpoint 不存在: {args_cli.checkpoint}')
        sys.exit(1)

    # 初始化 evaluator
    print('\n正在加载 EvaluatorMDMWrapper...')
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

    evaluator = EvaluatorMDMWrapper(
        't2m',
        device,
        build_dummy_args(),
        ckpt_path=args_cli.checkpoint
    )
    print('Evaluator 加载完成!\n')

    # 收集所有原始文件（不以 M 开头）
    all_files = sorted([
        f for f in os.listdir(args_cli.data_dir)
        if f.endswith('.npy') and not f.startswith('M')
    ])

    # 验证镜像文件存在
    valid_files = []
    for f in all_files:
        mirr_path = os.path.join(args_cli.data_dir, 'M' + f)
        if os.path.isfile(mirr_path):
            valid_files.append(f)

    skipped = len(all_files) - len(valid_files)
    if skipped > 0:
        print(f'警告: {skipped} 个文件缺少对应的镜像文件，已跳过')

    if args_cli.max_samples is not None:
        valid_files = valid_files[:args_cli.max_samples]

    print(f'共处理 {len(valid_files)} 对原始-镜像数据\n')

    # 主循环
    results = []

    for i, filename in enumerate(tqdm(valid_files, desc='计算相似度')):
        orig_path = os.path.join(args_cli.data_dir, filename)
        mirr_path = os.path.join(args_cli.data_dir, 'M' + filename)

        try:
            orig_len, mirr_len, cos_sim = compute_similarity(
                evaluator, orig_path, mirr_path
            )
        except Exception as e:
            tqdm.write(f'错误处理 {filename}: {e}')
            continue

        results.append({
            'filename': filename,
            'orig_len': orig_len,
            'mirr_len': mirr_len,
            'cos_sim': cos_sim,
        })

        if args_cli.verbose:
            tqdm.write(f'{filename}: orig_len={orig_len:>4d}, mirr_len={mirr_len:>4d}, '
                       f'cos_sim={cos_sim:.6f}')
        elif (i + 1) % args_cli.print_interval == 0:
            # 打印最近一批的摘要
            recent_sims = [r['cos_sim'] for r in results[-args_cli.print_interval:]]
            tqdm.write(f'  [{i+1:>6d}/{len(valid_files)}] '
                       f'最近 {len(recent_sims)} 个样本的平均相似度: {np.mean(recent_sims):.6f}')

    if len(results) == 0:
        print('没有成功处理任何数据对!')
        sys.exit(1)

    # 如果 verbose 模式下之前没打印过相似度，现在全部打印
    # (verbose 模式已在循环中打印)

    # 打印统计
    print_statistics(results)

    # 保存到文件
    if args_cli.output:
        output_path = args_cli.output
        with open(output_path, 'w') as f:
            f.write('filename,orig_len,mirr_len,cos_sim\n')
            for r in results:
                f.write(f'{r["filename"]},{r["orig_len"]},{r["mirr_len"]},{r["cos_sim"]:.6f}\n')
        print(f'\n详细结果已保存到: {output_path}')
    else:
        # 默认保存到 CSV
        default_output = 'test_mirror_similarity_results.csv'
        with open(default_output, 'w') as f:
            f.write('filename,orig_len,mirr_len,cos_sim\n')
            for r in results:
                f.write(f'{r["filename"]},{r["orig_len"]},{r["mirr_len"]},{r["cos_sim"]:.6f}\n')
        print(f'\n详细结果已保存到: {default_output}')


if __name__ == '__main__':
    main()
