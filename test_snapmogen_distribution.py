"""
对比两种 mean/std 的归一化效果：
  A. 官方：按特征组聚合 std（7 个组，组内共享一个 std 值）
  B. self：逐元素独立计算 mean/std（dataset/snapmogen_norm/）

目的：检查官方归一化是否会导致某些维度数值膨胀，影响扩散模型训练。
"""

import numpy as np
import glob
import os
from collections import OrderedDict

# ─── 配置 ─────────────────────────────────────────────────────────────
FEAT_DIR = '/data/motion/SnapMoGen/renamed_feats'
N_SAMPLE = 100  # 采样文件数

# 加载两种 mean/std
OFFICIAL_MEAN = np.load('/data/motion/SnapMoGen/meta_data/mean.npy')
OFFICIAL_STD  = np.load('/data/motion/SnapMoGen/meta_data/std.npy')
SELF_MEAN = np.load('/home/deli/project/reward_mdm/dataset/snapmogen_norm/mean.npy')
SELF_STD  = np.load('/home/deli/project/reward_mdm/dataset/snapmogen_norm/std.npy')

# 特征组定义
FEATURE_GROUPS = OrderedDict({
    'root_rot_velocity':    np.array([0]),
    'root_linear_velocity': np.array([1, 2]),
    'root_y':               np.array([3]),
    'joint_rotations':      np.arange(4, 148),    # 24 joints × 6
    'world_positions':      np.arange(148, 220),  # 24 joints × 3
    'local_velocities':     np.arange(220, 292),  # 24 joints × 3
    'foot_contacts':        np.arange(292, 296),
})

N_DIMS = 296


def get_group_name(dim):
    for name, dims in FEATURE_GROUPS.items():
        if dim in dims:
            rel_idx = int(dim - dims[0])
            return name, rel_idx
    return 'unknown', -1


def normalize(data, mean, std):
    """归一化，std=0 时用 1e-8 防止除零"""
    return (data - mean) / np.maximum(std, 1e-8)


def main():
    # ── 打印两种 std 的基本信息 ──
    print("=" * 70)
    print("  两种归一化参数对比")
    print("=" * 70)

    print(f"\n  官方 std（7 个特征组共享值）：")
    print(f"   唯一值: {np.sort(np.unique(OFFICIAL_STD))}")
    print(f"\n  逐元素 self std（每维独立）：")
    print(f"   范围: [{SELF_STD.min():.8f}, {SELF_STD.max():.4f}]")
    print(f"   std < 1e-4 的维度数: {(SELF_STD < 1e-4).sum()}")

    # 每组内 official std vs self 各维 std 的对比
    print(f"\n  {'Group':<25s} {'official_std':>12s} {'self_std_range':>32s} {'self_std_mean':>14s}")
    print(f"  {'-'*85}")
    for name, dims in FEATURE_GROUPS.items():
        o_std = OFFICIAL_STD[dims[0]]
        s_min, s_max = SELF_STD[dims].min(), SELF_STD[dims].max()
        s_mean = SELF_STD[dims].mean()
        print(f"  {name:<25s} {o_std:>12.6f}  [{s_min:.4f}, {s_max:.4f}]  {s_mean:>14.6f}")

    # ── 收集采样文件 ──
    all_files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))
    step = max(1, len(all_files) // N_SAMPLE)
    sample_files = all_files[::step][:N_SAMPLE]
    print(f"\n  总文件数: {len(all_files)}，采样: {len(sample_files)} 个文件")

    # ── 逐文件归一化并收集统计 ──
    # 记录每一帧每个维度的归一化值（聚合统计用）
    official_abs_max_per_dim = np.zeros(N_DIMS)  # 逐维最大绝对值
    self_abs_max_per_dim = np.zeros(N_DIMS)
    official_global_max = 0.0  # 全数据最大绝对值
    self_global_max = 0.0

    # 记录最大绝对值出现的具体文件和维度
    official_max_info = {'val': 0.0, 'dim': -1, 'file': '', 'frame': -1}
    self_max_info = {'val': 0.0, 'dim': -1, 'file': '', 'frame': -1}

    total_frames = 0

    print(f"\n  逐文件归一化...")
    for fi, fp in enumerate(sample_files):
        data = np.load(fp).astype(np.float64)  # (T, 296)
        basename = os.path.basename(fp)
        T = data.shape[0]
        total_frames += T

        # 官方归一化
        z_off = normalize(data, OFFICIAL_MEAN, OFFICIAL_STD)
        # self 归一化
        z_self = normalize(data, SELF_MEAN, SELF_STD)

        # 更新逐维最大绝对值
        off_abs = np.abs(z_off)
        self_abs = np.abs(z_self)
        official_abs_max_per_dim = np.maximum(official_abs_max_per_dim, off_abs.max(axis=0))
        self_abs_max_per_dim = np.maximum(self_abs_max_per_dim, self_abs.max(axis=0))

        # 检查全局最大值
        off_max_val = off_abs.max()
        off_max_frame, off_max_dim = np.unravel_index(off_abs.argmax(), off_abs.shape)
        if off_max_val > official_max_info['val']:
            official_max_info = {'val': off_max_val, 'dim': off_max_dim,
                                 'file': basename, 'frame': off_max_frame}
        official_global_max = max(official_global_max, off_max_val)

        self_max_val = self_abs.max()
        self_max_frame, self_max_dim = np.unravel_index(self_abs.argmax(), self_abs.shape)
        if self_max_val > self_max_info['val']:
            self_max_info = {'val': self_max_val, 'dim': self_max_dim,
                             'file': basename, 'frame': self_max_frame}
        self_global_max = max(self_global_max, self_max_val)

        if (fi + 1) % 20 == 0:
            print(f'    {fi+1}/{len(sample_files)} ...')

    # ── 打印结果 ──
    print(f"\n{'=' * 70}")
    print(f"  归一化后数值分布对比（{total_frames} 帧）")
    print(f"{'=' * 70}")

    print(f"\n  📊 全局最大绝对值:")
    print(f"    官方归一化: {official_max_info['val']:.2f} "
          f"(dim={official_max_info['dim']}, file={official_max_info['file']}, "
          f"frame={official_max_info['frame']})")
    gname, rel = get_group_name(official_max_info['dim'])
    print(f"        → 维度归属: {gname}, 组内偏移={rel}")
    print(f"    逐元素归一化: {self_max_info['val']:.2f} "
          f"(dim={self_max_info['dim']}, file={self_max_info['file']}, "
          f"frame={self_max_info['frame']})")
    gname, rel = get_group_name(self_max_info['dim'])
    print(f"        → 维度归属: {gname}, 组内偏移={rel}")

    # ── 逐维最大绝对值 TOP 维度 ──
    print(f"\n  📊 逐维最大绝对值 (|z|_max) TOP 20:")
    print(f"  {'dim':>6s}  {'group':<25s}  {'官方|z|max':>12s}  {'self|z|max':>12s}  {'膨胀比':>8s}")
    print(f"  {'-'*70}")

    # 按官方归一化的逐维最大值排序
    top_dims = np.argsort(official_abs_max_per_dim)[::-1][:20]
    for d in top_dims:
        gname, rel = get_group_name(d)
        ratio = official_abs_max_per_dim[d] / max(self_abs_max_per_dim[d], 1e-8)
        marker = ' ⚠️' if ratio > 1.5 else ''
        print(f"  [{d:3d}]  {gname:<25s}  {official_abs_max_per_dim[d]:>12.2f}  "
              f"{self_abs_max_per_dim[d]:>12.2f}  {ratio:>8.2f}x{marker}")

    # ── 按特征组统计膨胀比 ──
    print(f"\n  📊 按特征组的膨胀比 (官方|z|max / self|z|max):")
    print(f"  {'Group':<25s}  {'组内dim数':>8s}  {'官方|z|max':>12s}  "
          f"{'self|z|max':>12s}  {'膨胀比':>8s}  {'判定':>10s}")
    print(f"  {'-'*80}")

    for name, dims in FEATURE_GROUPS.items():
        o_max = official_abs_max_per_dim[dims].max()
        s_max = self_abs_max_per_dim[dims].max()
        ratio = o_max / max(s_max, 1e-8)
        if ratio > 2.0:
            verdict = '🔴 严重膨胀'
        elif ratio > 1.3:
            verdict = '🟡 轻微膨胀'
        else:
            verdict = '🟢 正常'
        print(f"  {name:<25s}  {len(dims):>8d}  {o_max:>12.2f}  "
              f"{s_max:>12.2f}  {ratio:>8.2f}x  {verdict}")

    # ── 值域分布 ──
    print(f"\n  📊 |z| 值域分布（所有维度所有帧混合）:")

    all_files_sample = sample_files
    off_all = []
    self_all = []

    # 只收集前 20 个文件的全量数据用于分位数统计（避免内存过大）
    for fp in all_files_sample[:20]:
        data = np.load(fp).astype(np.float64)
        off_all.append(np.abs(normalize(data, OFFICIAL_MEAN, OFFICIAL_STD)).ravel())
        self_all.append(np.abs(normalize(data, SELF_MEAN, SELF_STD)).ravel())
    off_all = np.concatenate(off_all)
    self_all = np.concatenate(self_all)

    print(f"  {'分位':<10s} {'官方|z|':>12s} {'self|z|':>12s}")
    for pct in [50, 90, 95, 99, 99.9, 100]:
        o_val = np.percentile(off_all, pct)
        s_val = np.percentile(self_all, pct)
        print(f"  {pct:5.1f}%   {o_val:>12.4f}  {s_val:>12.4f}")

    # ── 结论 ──
    print(f"\n{'=' * 70}")
    print(f"  结论")
    print(f"{'=' * 70}")

    # 计算超标维度的比例
    off_bad = (official_abs_max_per_dim > 10.0).sum()
    self_bad = (self_abs_max_per_dim > 10.0).sum()
    print(f"  |z|max > 10  的维度数: 官方={off_bad}, self={self_bad}")
    off_bad20 = (official_abs_max_per_dim > 20.0).sum()
    self_bad20 = (self_abs_max_per_dim > 20.0).sum()
    print(f"  |z|max > 20  的维度数: 官方={off_bad20}, self={self_bad20}")

    # 关键发现
    print(f"\n  关键发现:")
    if official_global_max > 50:
        print(f"  ⚠️ 官方归一化后存在极端值 (|z|max={official_global_max:.1f})，")
        print(f"     这可能导致扩散模型训练时梯度不稳定。")
    else:
        print(f"  官方归一化后的最大值 ({official_global_max:.1f}) 在可接受范围内。")

    # 膨胀比大的维度分析
    inflation_ratio = official_abs_max_per_dim / np.maximum(self_abs_max_per_dim, 1e-8)
    high_inflation = np.where(inflation_ratio > 1.3)[0]
    if len(high_inflation) > 0:
        print(f"\n  膨胀比 > 1.3x 的维度 ({len(high_inflation)} 个):")
        for d in high_inflation[:15]:
            gname, rel = get_group_name(d)
            print(f"    dim[{d:3d}] {gname}[{rel}]: "
                  f"官方|z|max={official_abs_max_per_dim[d]:.1f}, "
                  f"self|z|max={self_abs_max_per_dim[d]:.1f}, "
                  f"ratio={inflation_ratio[d]:.2f}x")


if __name__ == '__main__':
    main()
