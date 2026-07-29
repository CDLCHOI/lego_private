"""
统计 HumanML3D 和 SnapMoGen 两个数据集 motion npy 文件的数值分布特征。

HumanML3D: /data/motion/HumanML3D/new_joint_vecs/*.npy, dim=263, float32
SnapMoGen:  /data/motion/SnapMoGen/renamed_feats/*.npy,    dim=296, float64
"""

import numpy as np
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def compute_file_stats(filepath):
    """计算单个 npy 文件的统计量"""
    data = np.load(filepath).astype(np.float64)  # 统一转为 float64 避免溢出
    flat = data.ravel()
    return {
        'path': filepath,
        'shape': data.shape,
        'frames': data.shape[0],
        'dim': data.shape[1],
        'min': flat.min(),
        'max': flat.max(),
        'mean': flat.mean(),
        'std': flat.std(),
        'total_values': flat.size,
    }


def compute_per_dimension_stats(all_data_list, dataset_name, n_dims):
    """
    分批加载所有数据，按维度（列）计算 min, max, mean, std。
    使用在线算法 (Welford) 避免一次性加载所有数据到内存。
    """
    # 用于在线均值和方差计算
    dim_count = np.zeros(n_dims, dtype=np.float64)
    dim_mean = np.zeros(n_dims, dtype=np.float64)
    dim_m2 = np.zeros(n_dims, dtype=np.float64)
    dim_min = np.full(n_dims, np.inf, dtype=np.float64)
    dim_max = np.full(n_dims, -np.inf, dtype=np.float64)

    print(f"\n[{dataset_name}] 逐维度统计 (共 {n_dims} 维)...")
    for filepath in tqdm(all_data_list, desc=dataset_name):
        data = np.load(filepath).astype(np.float64)  # (L, D)
        L = data.shape[0]

        for d in range(n_dims):
            col = data[:, d]
            col_min = col.min()
            col_max = col.max()
            col_mean = col.mean()

            if col_min < dim_min[d]:
                dim_min[d] = col_min
            if col_max > dim_max[d]:
                dim_max[d] = col_max

            # Welford 在线算法：合并当前列的统计量
            # 将当前文件的列统计合并到全局统计中
            n_a = dim_count[d]
            n_b = L
            if n_b == 0:
                continue
            delta = col_mean - dim_mean[d]
            dim_mean[d] = (n_a * dim_mean[d] + n_b * col_mean) / (n_a + n_b)
            dim_m2[d] += ((col - col_mean) ** 2).sum() + delta ** 2 * n_a * n_b / (n_a + n_b)
            dim_count[d] += n_b

    # 计算 per-dimension std
    dim_std = np.sqrt(dim_m2 / dim_count)

    return {
        'dim_min': dim_min,
        'dim_max': dim_max,
        'dim_mean': dim_mean,
        'dim_std': dim_std,
    }


def print_summary(name, file_stats_list, per_dim_stats, n_dims):
    """打印汇总统计"""
    print(f"\n{'=' * 70}")
    print(f"  {name} 数据集统计")
    print(f"{'=' * 70}")

    # 文件级别统计
    num_files = len(file_stats_list)
    total_frames = sum(s['frames'] for s in file_stats_list)
    total_values = sum(s['total_values'] for s in file_stats_list)

    print(f"\n  📁 文件数量: {num_files}")
    print(f"  🎬 总帧数:   {total_frames}")
    print(f"  📐 总数值量: {total_values:,}  ({total_values / 1e9:.2f} G)")
    print(f"  📏 维度数:   {n_dims}")

    # 全量数据统计
    global_min = min(s['min'] for s in file_stats_list)
    global_max = max(s['max'] for s in file_stats_list)

    # 将所有元素的 mean/std 用加权方式合并（每个文件的 total_values 加权）
    total_vals = sum(s['total_values'] for s in file_stats_list)
    global_mean = sum(s['mean'] * s['total_values'] for s in file_stats_list) / total_vals if total_vals > 0 else 0.0
    global_var = sum((s['std'] ** 2 * (s['total_values'] - 1) +
                      s['total_values'] * (s['mean'] - global_mean) ** 2)
                     for s in file_stats_list) / (total_vals - 1) if total_vals > 1 else 0.0
    global_std = np.sqrt(global_var)

    # 百分位数：收集所有文件的最小值、最大值
    all_mins = np.array([s['min'] for s in file_stats_list])
    all_maxs = np.array([s['max'] for s in file_stats_list])
    all_means = np.array([s['mean'] for s in file_stats_list])
    all_stds = np.array([s['std'] for s in file_stats_list])
    all_frames = np.array([s['frames'] for s in file_stats_list])

    print(f"\n  📊 全量数据全局统计:")
    print(f"     Min:  {global_min:>12.6f}")
    print(f"     Max:  {global_max:>12.6f}")
    print(f"     Mean: {global_mean:>12.6f}")
    print(f"     Std:  {global_std:>12.6f}")

    print(f"\n  📊 文件级别统计:")
    print(f"     每文件帧数 Min/Max:  {all_frames.min():.0f} / {all_frames.max():.0f}")
    print(f"     每文件帧数 Mean/Std: {all_frames.mean():.1f} / {all_frames.std():.1f}")
    print(f"     每文件 mean 值范围:   [{all_means.min():.6f}, {all_means.max():.6f}]")
    print(f"     每文件 std 值范围:    [{all_stds.min():.6f}, {all_stds.max():.6f}]")
    print(f"     每文件 min 值范围:    [{all_mins.min():.6f}, {all_mins.max():.6f}]")
    print(f"     每文件 max 值范围:    [{all_maxs.min():.6f}, {all_maxs.max():.6f}]")

    # 维度级别统计
    if per_dim_stats is not None:
        pd = per_dim_stats
        print(f"\n  📐 逐维度统计 (共 {n_dims} 维):")
        print(f"     维度 min 的范围:  [{pd['dim_min'].min():.6f}, {pd['dim_min'].max():.6f}]")
        print(f"     维度 max 的范围:  [{pd['dim_max'].min():.6f}, {pd['dim_max'].max():.6f}]")
        print(f"     维度 mean 的范围: [{pd['dim_mean'].min():.6f}, {pd['dim_mean'].max():.6f}]")
        print(f"     维度 std 的范围:  [{pd['dim_std'].min():.6f}, {pd['dim_std'].max():.6f}]")

        # 前5个和后5个维度的详细统计
        print(f"\n  📐 前 5 维详细统计:")
        for d in range(min(5, n_dims)):
            print(f"     dim[{d:3d}]: min={pd['dim_min'][d]:>12.6f}  max={pd['dim_max'][d]:>12.6f}  "
                  f"mean={pd['dim_mean'][d]:>12.6f}  std={pd['dim_std'][d]:>12.6f}")

        print(f"\n  📐 后 5 维详细统计:")
        for d in range(max(0, n_dims - 5), n_dims):
            print(f"     dim[{d:3d}]: min={pd['dim_min'][d]:>12.6f}  max={pd['dim_max'][d]:>12.6f}  "
                  f"mean={pd['dim_mean'][d]:>12.6f}  std={pd['dim_std'][d]:>12.6f}")

        # 找出极值最夸张的前10维
        range_per_dim = pd['dim_max'] - pd['dim_min']
        top_range_idx = np.argsort(range_per_dim)[::-1][:10]
        print(f"\n  🔥 数值范围 (max-min) 最大的 10 个维度:")
        for idx in top_range_idx:
            print(f"     dim[{idx:3d}]: range={range_per_dim[idx]:.6f}  "
                  f"min={pd['dim_min'][idx]:.6f}  max={pd['dim_max'][idx]:.6f}  "
                  f"mean={pd['dim_mean'][idx]:.6f}  std={pd['dim_std'][idx]:.6f}")

        # 找出 std 最大的前10维
        top_std_idx = np.argsort(pd['dim_std'])[::-1][:10]
        print(f"\n  🔥 标准差最大的 10 个维度:")
        for idx in top_std_idx:
            print(f"     dim[{idx:3d}]: std={pd['dim_std'][idx]:.6f}  "
                  f"min={pd['dim_min'][idx]:.6f}  max={pd['dim_max'][idx]:.6f}  "
                  f"mean={pd['dim_mean'][idx]:.6f}")


def main():
    # ── 路径配置 ──
    # HumanML3D 数据实际位于 /data/motion/HumanML3D/，项目中的 ./dataset/HumanML3D/ 没有 npy 文件
    humanml_dir = '/data/motion/HumanML3D/new_joint_vecs'
    snapmogen_dir = '/data/motion/SnapMoGen/renamed_feats'

    # ── 收集所有文件 ──
    print("🔍 扫描文件...")
    humanml_files = sorted(glob(os.path.join(humanml_dir, '*.npy')))
    snapmogen_files = sorted(glob(os.path.join(snapmogen_dir, '*.npy')))
    print(f"  HumanML3D: {len(humanml_files)} 个文件")
    print(f"  SnapMoGen: {len(snapmogen_files)} 个文件")

    # ── HumanML3D 逐文件统计 ──
    print("\n" + "=" * 70)
    print("  处理 HumanML3D 数据集...")
    print("=" * 70)
    hml_stats = []
    for fp in tqdm(humanml_files, desc='HumanML3D'):
        hml_stats.append(compute_file_stats(fp))

    # ── SnapMoGen 逐文件统计 ──
    print("\n" + "=" * 70)
    print("  处理 SnapMoGen 数据集...")
    print("=" * 70)
    snap_stats = []
    for fp in tqdm(snapmogen_files, desc='SnapMoGen'):
        snap_stats.append(compute_file_stats(fp))

    # ── 逐维度统计 ──
    hml_dim_stats = compute_per_dimension_stats(humanml_files, 'HumanML3D', n_dims=263)
    snap_dim_stats = compute_per_dimension_stats(snapmogen_files, 'SnapMoGen', n_dims=296)

    # ── 打印汇总 ──
    print_summary('HumanML3D', hml_stats, hml_dim_stats, n_dims=263)
    print_summary('SnapMoGen', snap_stats, snap_dim_stats, n_dims=296)

    # ── 对比分析 ──
    print(f"\n{'=' * 70}")
    print(f"  🔬 两个数据集对比分析")
    print(f"{'=' * 70}")

    hml_total = sum(s['total_values'] for s in hml_stats)
    snap_total = sum(s['total_values'] for s in snap_stats)
    hml_mean = sum(s['mean'] * s['total_values'] for s in hml_stats) / hml_total
    snap_mean = sum(s['mean'] * s['total_values'] for s in snap_stats) / snap_total

    print(f"\n  📊 全局均值对比:")
    print(f"     HumanML3D 全局均值: {hml_mean:.6f}")
    print(f"     SnapMoGen  全局均值: {snap_mean:.6f}")

    # 值域对比
    print(f"\n  📊 值域对比:")
    print(f"     HumanML3D 值域: [{min(s['min'] for s in hml_stats):.4f}, "
          f"{max(s['max'] for s in hml_stats):.4f}]")
    print(f"     SnapMoGen  值域: [{min(s['min'] for s in snap_stats):.4f}, "
          f"{max(s['max'] for s in snap_stats):.4f}]")

    # 维度方差分布对比
    print(f"\n  📊 维度均值分布对比:")
    print(f"     HumanML3D 维度 mean 范围: [{hml_dim_stats['dim_mean'].min():.6f}, "
          f"{hml_dim_stats['dim_mean'].max():.6f}]")
    print(f"     SnapMoGen  维度 mean 范围: [{snap_dim_stats['dim_mean'].min():.6f}, "
          f"{snap_dim_stats['dim_mean'].max():.6f}]")

    print(f"\n  📊 维度标准差分布对比:")
    print(f"     HumanML3D 维度 std 范围: [{hml_dim_stats['dim_std'].min():.6f}, "
          f"{hml_dim_stats['dim_std'].max():.6f}]")
    print(f"     SnapMoGen  维度 std 范围: [{snap_dim_stats['dim_std'].min():.6f}, "
          f"{snap_dim_stats['dim_std'].max():.6f}]")

    print(f"\n  📊 维度值域对比:")
    print(f"     HumanML3D dim_min 范围: [{hml_dim_stats['dim_min'].min():.6f}, "
          f"{hml_dim_stats['dim_min'].max():.6f}]")
    print(f"     SnapMoGen  dim_min 范围: [{snap_dim_stats['dim_min'].min():.6f}, "
          f"{snap_dim_stats['dim_min'].max():.6f}]")
    print(f"     HumanML3D dim_max 范围: [{hml_dim_stats['dim_max'].min():.6f}, "
          f"{hml_dim_stats['dim_max'].max():.6f}]")
    print(f"     SnapMoGen  dim_max 范围: [{snap_dim_stats['dim_max'].min():.6f}, "
          f"{snap_dim_stats['dim_max'].max():.6f}]")

    # 看一下 HumanML3D 的前4维（root velocity 信息）
    print(f"\n  📐 HumanML3D 前4维 (root velocity):")
    for d in range(4):
        print(f"     dim[{d}]: min={hml_dim_stats['dim_min'][d]:.6f}  max={hml_dim_stats['dim_max'][d]:.6f}  "
              f"mean={hml_dim_stats['dim_mean'][d]:.6f}  std={hml_dim_stats['dim_std'][d]:.6f}")

    # HumanML3D 后3维 (feet contact)
    print(f"\n  📐 HumanML3D 后3维 (feet contact):")
    for d in range(260, 263):
        print(f"     dim[{d}]: min={hml_dim_stats['dim_min'][d]:.6f}  max={hml_dim_stats['dim_max'][d]:.6f}  "
              f"mean={hml_dim_stats['dim_mean'][d]:.6f}  std={hml_dim_stats['dim_std'][d]:.6f}")

    print(f"\n✅ 统计完成!")


if __name__ == '__main__':
    main()
