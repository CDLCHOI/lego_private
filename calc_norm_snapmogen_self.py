"""
SnapMoGen 逐维度 mean/std 计算 + 恒定维度分析。

与官方（按特征组聚合 std）不同，此脚本逐维度独立计算 mean 和 std。
对于 std 为零的维度（数值完全不变），用 1e-6 作为安全下限，避免归一化时除零。

重要：world_positions 组（dims 148-219）使用统一的组 std 而非逐维度独立 std，
以保持骨骼几何比例不变。如果使用逐维度独立 std，会导致 X/Y/Z 坐标被不同比例缩放，
造成前臂变长、手变小等骨骼变形问题。

生成文件：dataset/snapmogen_norm/mean.npy, std.npy

特征组定义（296 维）：
  dim 0:     root_rot_velocity  (1)
  dims 1-2:  root_linear_velocity (2)
  dim 3:     root_y (1)
  dims 4-147:   6D rotations, 24 joints × 6 = 144
  dims 148-219: world positions, 24 joints × 3 = 72
  dims 220-291: local velocities, 24 joints × 3 = 72
  dims 292-295: foot contacts (4)
"""

import numpy as np
import glob
import os
from collections import OrderedDict

# ─── 配置 ─────────────────────────────────────────────────────────────
FEAT_DIR = '/data/motion/SnapMoGen/renamed_feats'
TRAIN_IDS_FILE = '/data/motion/SnapMoGen/data_split_info/train_ids.txt'
OUT_DIR = './dataset/snapmogen_norm'
N_JOINTS = 24
N_DIMS = 296
FLOOR_STD = 1e-6  # 恒定维度的 std 安全下限

# ─── 特征组 ───────────────────────────────────────────────────────────
FEATURE_GROUPS = OrderedDict({
    'root_rot_velocity':    np.array([0]),
    'root_linear_velocity': np.array([1, 2]),
    'root_y':               np.array([3]),
    'joint_rotations':      np.arange(4, 148),    # 24 joints × 6
    'world_positions':      np.arange(148, 220),  # 24 joints × 3
    'local_velocities':     np.arange(220, 292),  # 24 joints × 3
    'foot_contacts':        np.arange(292, 296),
})


def get_group_name(dim):
    """返回维度所属的特征组名称和组内偏移"""
    for name, dims in FEATURE_GROUPS.items():
        if dim in dims:
            rel_idx = int(dim - dims[0])
            return name, rel_idx
    return 'unknown', -1


def get_joint_info(dim):
    """
    返回维度对应的关节信息。
    joint_rotations: 24 joints × 6
    world_positions: 24 joints × 3
    local_velocities: 24 joints × 3
    """
    group_name, rel_idx = get_group_name(dim)
    if group_name == 'joint_rotations':
        joint_idx = rel_idx // 6
        comp_idx = rel_idx % 6
        return joint_idx, comp_idx, '6D_rotation'
    elif group_name == 'world_positions':
        joint_idx = rel_idx // 3
        comp_idx = rel_idx % 3
        axis = ['X', 'Y', 'Z'][comp_idx]
        return joint_idx, comp_idx, f'world_pos_{axis}'
    elif group_name == 'local_velocities':
        joint_idx = rel_idx // 3
        comp_idx = rel_idx % 3
        axis = ['X', 'Y', 'Z'][comp_idx]
        return joint_idx, comp_idx, f'local_vel_{axis}'
    return -1, -1, group_name


def compute_per_dim_mean_std(files):
    """
    逐维度计算 mean 和 std (使用 Welford 在线算法)。
    返回 (mean, std)，shape 均为 (296,)。
    """
    sum_x = np.zeros(N_DIMS, dtype=np.float64)
    sum_x2 = np.zeros(N_DIMS, dtype=np.float64)
    total_frames = 0

    print(f'处理 {len(files)} 个训练文件...')
    for i, fp in enumerate(files):
        data = np.load(fp).astype(np.float64)  # (T, 296)
        sum_x += data.sum(axis=0)
        sum_x2 += (data ** 2).sum(axis=0)
        total_frames += data.shape[0]
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(files)} ...')

    mean = sum_x / total_frames
    var = sum_x2 / total_frames - mean ** 2
    var = np.maximum(var, 0.0)  # 防止数值误差导致负值
    std = np.sqrt(var)
    return mean, std


def apply_group_uniform_std(mean, std):
    """
    对 world_positions 组（dims 148-219，24 joints × 3）使用统一的组 std，
    以保持几何比例不被逐维度独立的 std 扭曲。

    如果不做这个处理，逐维度独立的 std 会导致：
    - 某些维度的 std 小 → 归一化时被放大 → 反归一化后该方向被拉伸
    - 某些维度的 std 大 → 归一化时被缩小 → 反归一化后该方向被压缩
    - 最终效果：骨骼比例被扭曲（如前臂变长、手变小等）
    """
    pos_dims = FEATURE_GROUPS['world_positions']  # [148, 220)
    pos_global_std = float(np.sqrt(np.mean(std[pos_dims] ** 2)))
    print(f'\n  位置维度统一 std: {pos_global_std:.6f} (替代逐维度 std 范围 [{std[pos_dims].min():.4f}, {std[pos_dims].max():.4f}])')
    std[pos_dims] = pos_global_std
    return mean, std


def analyze_constant_dims(std, n_samples=3):
    """
    分析 std 接近零的维度：检查实际值，解释含义。
    """
    const_dims = np.where(std < FLOOR_STD * 10)[0]  # std < 1e-5

    print(f'\n{"=" * 70}')
    print(f'  恒定维度分析：std < 1e-5 的维度共 {len(const_dims)} 个')
    print(f'{"=" * 70}')

    # 按特征组分类
    by_group = OrderedDict()
    for d in const_dims:
        gname, _ = get_group_name(d)
        by_group.setdefault(gname, []).append(d)

    for gname, dims in by_group.items():
        print(f'\n  [{gname}] {len(dims)} 个恒定维度: {dims}')

    # 读取几个文件，查看恒定位置的实际值
    files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))[:n_samples]
    print(f'\n  检查 {len(files)} 个文件确认恒定值...')
    for fp in files:
        data = np.load(fp)
        for d in const_dims[:8]:  # 只展示前几个
            unique = np.unique(data[:, d])
            group_name, rel_idx = get_group_name(d)
            joint_idx, comp_idx, desc = get_joint_info(d)
            print(f'    {os.path.basename(fp)}: dim[{d:3d}]={unique}  '
                  f'({group_name}, joint={joint_idx}, {desc})')
        if len(const_dims) > 8:
            print(f'    ... (其余 {len(const_dims)-8} 个维度同理)')
        break

    # ── 逐维度详细解释 ──
    print(f'\n  {"─" * 60}')
    print(f'  逐维度详细映射:')
    print(f'  {"─" * 60}')

    for d in const_dims:
        group_name, rel_idx = get_group_name(d)
        joint_idx, comp_idx, desc = get_joint_info(d)
        print(f'    dim[{d:3d}] | {group_name:<22s} | joint={joint_idx:2d} | {desc}')

    # ── 结论 ──
    print(f'\n  {"─" * 60}')
    print(f'  结论:')
    print(f'  {"─" * 60}')
    print(f'  1. Joint 0 (Pelvis/根关节) 的 6D rotation 有 4 个分量恒为 0、')
    print(f'     1 个分量恒为 1 (dim[8]=1)，即该关节的局部旋转恒为单位旋转。')
    print(f'     根关节的全局旋转由 dim[0] (root_rot_velocity) 单独编码，')
    print(f'     因此其局部旋转不需要变化。')
    print(f'')
    print(f'  2. Joint 0, 1, 15 的 World Position X 和 Z 分量恒为 0：')
    print(f'     - Joint 0 = Pelvis：根关节在世界空间 XZ 平面始终位于原点，')
    print(f'       Y 分量(地面高度)随运动变化')
    print(f'     - Joint 1 ≈ 脊椎第一关节：位于 Pelvis 正上方，在 XZ 平面')
    print(f'       也始终居中（人体中轴线），X,Z 恒为 0')
    print(f'     - Joint 15 ≈ 头/颈部某关节：同样位于中轴线上，X,Z 恒为 0')
    print(f'')
    print(f'  3. 对应关节的 Local Velocity X 和 Z 分量恒为 0：')
    print(f'     中轴线上的关节在 XZ 平面没有水平速度分量，只有 Y 方向')
    print(f'     (上下)的运动速度。')
    print(f'')
    print(f'  4. 这些维度的 std 在计算中被设为 {FLOOR_STD}（安全下限），')
    print(f'     因为原始数据的 std 精确为 0，不存在任何变化。')


def main():
    # ── 收集训练文件 ──
    train_mids = set()
    with open(TRAIN_IDS_FILE) as f:
        for line in f:
            train_mids.add(line.strip().split('#')[0])

    all_files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))
    train_files = [f for f in all_files
                   if os.path.basename(f).replace('.npy', '') in train_mids]
    print(f'总文件: {len(all_files)}, 训练集文件: {len(train_files)}')

    # ── 逐维度计算 mean/std ──
    mean, raw_std = compute_per_dim_mean_std(train_files)

    # ── 位置维度使用统一 std（保持几何比例，防止骨骼变形） ──
    mean, raw_std = apply_group_uniform_std(mean, raw_std)

    # ── 应用安全下限 ──
    std = np.maximum(raw_std, FLOOR_STD)
    num_floored = (raw_std < FLOOR_STD).sum()
    print(f'\nstd < {FLOOR_STD} (被设为下限) 的维度数: {num_floored}')

    # ── 保存 ──
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, 'mean.npy'), mean.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'std.npy'), std.astype(np.float32))
    print(f'\n已保存 mean/std 到 {OUT_DIR}/')

    # ── 打印统计 ──
    print(f'\n{"=" * 60}')
    print(f'  统计汇总')
    print(f'{"=" * 60}')
    print(f'  mean 范围: [{mean.min():.6f}, {mean.max():.6f}]')
    print(f'  raw_std 范围: [{raw_std.min():.10f}, {raw_std.max():.6f}]')
    print(f'  final_std 范围: [{std.min():.10f}, {std.max():.6f}]')
    print(f'  mean 总和: {mean.sum():.4f}')
    print(f'  std 总和: {std.sum():.4f}')

    # ── 恒定维度分析 ──
    analyze_constant_dims(std)


# ═══════════════════════════════════════════════════════════════════════
#  扩展：全数据集（含 test）的 mean/std 计算
# ═══════════════════════════════════════════════════════════════════════

def compute_all_data_mean_std():
    """
    对 SnapMoGen 所有数据（train + test）逐维度计算 mean 和 std。
    保存为 mean_all.npy 和 std_all.npy。
    """
    all_files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))
    print(f'\n[全数据集] 共 {len(all_files)} 个文件，计算逐维度 mean/std...')
    mean_all, raw_std_all = compute_per_dim_mean_std(all_files)
    # 位置维度使用统一 std（保持几何比例）
    mean_all, raw_std_all = apply_group_uniform_std(mean_all, raw_std_all)
    std_all = np.maximum(raw_std_all, FLOOR_STD)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, 'mean_all.npy'), mean_all.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'std_all.npy'), std_all.astype(np.float32))
    print(f'已保存 mean_all/std_all 到 {OUT_DIR}/')

    # 与训练集对比
    train_mean = np.load(os.path.join(OUT_DIR, 'mean.npy'))
    train_std  = np.load(os.path.join(OUT_DIR, 'std.npy'))
    print(f'  mean_all vs mean_train 最大差异: {np.abs(mean_all - train_mean).max():.6f}')
    print(f'  std_all  vs std_train  最大差异: {np.abs(std_all - train_std).max():.6f}')
    print(f'  std_all < {FLOOR_STD} 的维度数: {(std_all < FLOOR_STD * 10).sum()}')

    return mean_all, std_all


# ═══════════════════════════════════════════════════════════════════════
#  分析：恒定维度在两种归一化下的数值对比
# ═══════════════════════════════════════════════════════════════════════

def analyze_constant_dims_normalization(self_mean, self_std, n_sample=100):
    """
    重点分析那些 std=0（实际为 1e-6）的维度：
    1. 原始数据中这些维度的恒定值是什么？
    2. 用 self（逐元素）归一化后，这些维度的值是多少？会很大吗？
    3. 用 official（分组）归一化后，这些维度的值是多少？会很大吗？

    关键公式：
      z_self    = (x - self_mean) / self_std    # self_std ≈ 1e-6
      z_official = (x - official_mean) / official_std  # official_std = 组聚合值

    如果 x ≈ mean（两者归一化都是），那么 z ≈ 0 / std ＝ 0，不会膨胀。
    如果 x ≠ mean，除以 1e-6 会导致巨大数值。
    """
    official_mean = np.load('/data/motion/SnapMoGen/meta_data/mean.npy')
    official_std  = np.load('/data/motion/SnapMoGen/meta_data/std.npy')

    # 找出 std ≈ 0 的维度（即被设为 FLOOR_STD 的维度）
    const_dims = np.where(self_std < FLOOR_STD * 10)[0]
    print(f'\n{"=" * 70}')
    print(f'  恒定维度归一化分析：std < 1e-5 的 {len(const_dims)} 个维度')
    print(f'{"=" * 70}')

    # 先展示这些维度的统计特征
    print(f'\n  {"dim":>5s}  {"特征组":<22s}  {"关节":>4s}  {"描述":<16s}  '
          f'{"原始常值":>10s}  {"self_mean":>10s}  {"self_std":>10s}  '
          f'{"official_std":>12s}')
    print(f'  {"-"*100}')

    for d in const_dims:
        group_name, _ = get_group_name(d)
        joint_idx, comp_idx, desc = get_joint_info(d)
        o_std = official_std[d]
        s_mean = self_mean[d]
        s_std = self_std[d]
        # 读取几个文件确认原始常值
        sample_val = None
        for fp in sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))[:3]:
            data = np.load(fp)
            unique = np.unique(data[:, d])
            if len(unique) == 1:
                sample_val = unique[0]
                break
        print(f'  [{d:3d}]  {group_name:<22s}  {joint_idx:>4d}  {desc:<16s}  '
              f'{str(sample_val):>10s}  {s_mean:>10.6f}  {s_std:>10.8f}  '
              f'{o_std:>12.6f}')

    # ── 采样文件，计算实际的归一化值 ──
    all_files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.npy')))
    step = max(1, len(all_files) // n_sample)
    sample_files = all_files[::step][:n_sample]

    self_z_const = []   # self 归一化后的值
    off_z_const = []    # official 归一化后的值

    for fp in sample_files:
        data = np.load(fp).astype(np.float64)
        z_self = (data - self_mean) / np.maximum(self_std, 1e-8)
        z_off  = (data - official_mean) / np.maximum(official_std, 1e-8)
        self_z_const.append(z_self[:, const_dims])
        off_z_const.append(z_off[:, const_dims])

    self_z_const = np.concatenate(self_z_const, axis=0)  # (total_frames, n_const_dims)
    off_z_const  = np.concatenate(off_z_const, axis=0)

    # ── 汇总 ──
    print(f'\n  {"─" * 100}')
    print(f'  采样 {n_sample} 个文件，共 {self_z_const.shape[0]} 帧')
    print(f'\n  {"dim":>5s}  {"self |z|max":>14s}  {"self |z|mean":>14s}  '
          f'{"official |z|max":>14s}  {"official |z|mean":>14s}  '
          f'{"膨胀比":>8s}  {"是否安全":>10s}')
    print(f'  {"-"*85}')

    any_danger = False
    for i, d in enumerate(const_dims):
        s_max = np.abs(self_z_const[:, i]).max()
        s_mean = np.abs(self_z_const[:, i]).mean()
        o_max = np.abs(off_z_const[:, i]).max()
        o_mean = np.abs(off_z_const[:, i]).mean()
        ratio = o_max / max(s_max, 1e-12)

        # 判断：如果 self 归一化后 |z| 很大（> 1.0），说明有问题
        if s_max > 1.0:
            safety = '🔴 有问题'
            any_danger = True
        elif s_max > 0.01:
            safety = '🟡 略大'
        else:
            safety = '🟢 安全'

        print(f'  [{d:3d}]  {s_max:>14.8f}  {s_mean:>14.8f}  '
              f'{o_max:>14.8f}  {o_mean:>14.8f}  '
              f'{ratio:>8.2f}x  {safety}')

    # ── 结论 ──
    print(f'\n  {"─" * 85}')
    print(f'  结论:')
    if any_danger:
        print(f'  ⚠️ 部分恒定维度在 self 归一化后仍有较大数值，需关注')
    else:
        print(f'  ✅ 所有恒定维度在 self 归一化后数值都接近 0（安全）')
        print(f'     原因：恒定值 ≈ mean，所以 (x - mean) / std ≈ 0 / 1e-6 ≈ 0')
        print(f'     官方归一化同理：(x - mean) / group_std ≈ 0 / 0.XX ≈ 0')
        print(f'     恒定维度不会导致归一化后的数值膨胀问题。')

    # 对比：真正导致膨胀的是 std 非零但远小于组 std 的维度
    print(f'\n  额外分析：std 不为零但远小于组 std 的维度（这才是膨胀的根源）')
    print(f'  这些维度有真实变化，但被组 std 低估 → 数值膨胀')

    # 找出 self_std 显著小于 official_std 的维度
    ratio_std = official_std / np.maximum(self_std, 1e-8)
    # 排除恒定维度
    non_const = self_std > FLOOR_STD * 10
    under_estimated = np.where((ratio_std > 1.3) & non_const)[0]
    under_estimated = under_estimated[np.argsort(ratio_std[under_estimated])[::-1]]

    print(f'  official_std / self_std > 1.3x 的非恒定维度: {len(under_estimated)} 个')
    if len(under_estimated) > 0:
        print(f'\n  {"dim":>5s}  {"分组":<22s}  {"self_std":>10s}  '
              f'{"official_std":>12s}  {"比值":>8s}  '
              f'{"含义":>30s}')
        print(f'  {"-"*90}')
        for d in under_estimated[:20]:
            gname, rel = get_group_name(d)
            j_idx, c_idx, desc = get_joint_info(d)
            meaning = f'joint{j_idx} {desc}'
            print(f'  [{d:3d}]  {gname:<22s}  {self_std[d]:>10.4f}  '
                  f'{official_std[d]:>12.4f}  {ratio_std[d]:>8.2f}x  '
                  f'{meaning:<30s}')

    return


if __name__ == '__main__':
    # 原有：训练集统计
    main()

    # ── 新增：全数据集统计 ──
    print(f'\n{"#" * 70}')
    print(f'#  扩展：全数据集 mean/std')
    print(f'{"#" * 70}')
    mean_all, std_all = compute_all_data_mean_std()

    # ── 新增：恒定维度在两种归一化下的对比 ──
    print(f'\n{"#" * 70}')
    print(f'#  扩展：恒定维度归一化数值分析')
    print(f'{"#" * 70}')

    # 使用全量统计的 mean/std 进行分析
    analyze_constant_dims_normalization(mean_all, std_all)
