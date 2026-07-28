"""重现官方 mean/std 的计算逻辑：逐维算 mean，按特征组聚合 std"""
import numpy as np, glob, os

feat_dir = '/data/motion/SnapMoGen/renamed_feats'
train_ids_file = '/data/motion/SnapMoGen/data_split_info/train_ids.txt'
feat_bias = 5.0
n_joints = 24

# ── 特征组定义 ──
# dim 0:     root_rot_velocity  (1)
# dims 1-2:  root_linear_velocity (2)
# dim 3:     root_y (1)
# dims 4-147:   6D rotations, 24 joints × 6 = 144
# dims 148-219: world positions, 24 joints × 3 = 72
# dims 220-291: local velocities, 24 joints × 3 = 72
# dims 292-295: foot contacts (4)

groups = {
    'root_rot_velocity':   (np.array([0]),       feat_bias),
    'root_linear_velocity':(np.array([1, 2]),    feat_bias),
    'root_y':              (np.array([3]),       feat_bias),
    'joint_rotations':     (np.arange(4, 148),   1.0),
    'world_positions':     (np.arange(148, 220), 1.0),
    'local_velocities':    (np.arange(220, 292), 1.0),
    'foot_contacts':       (np.arange(292, 296), feat_bias),
}

# ── 加载训练数据并逐维计算 ──
train_mids = set()
with open(train_ids_file) as f:
    for line in f:
        train_mids.add(line.strip().split('#')[0])

all_files = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))
train_files = [f for f in all_files if os.path.basename(f).replace('.npy','') in train_mids]
print(f'训练文件数: {len(train_files)}')

n_dims = 296
sum_x = np.zeros(n_dims); sum_x2 = np.zeros(n_dims); count = 0
for f in train_files:
    data = np.load(f)
    sum_x += data.sum(axis=0); sum_x2 += (data**2).sum(axis=0); count += data.shape[0]

computed_mean = sum_x / count
computed_std  = np.sqrt(np.maximum(sum_x2 / count - computed_mean**2, 1e-12))

# ── 按组聚合 std：组内 RMS，然后除以 feat_bias ──
repro_mean = computed_mean.copy()
repro_std  = np.zeros(n_dims)

for name, (dims, bias) in groups.items():
    rms = np.sqrt(np.mean(computed_std[dims] ** 2))  # 组内 RMS
    scaled = rms / bias
    repro_std[dims] = scaled
    print(f'{name:<25s}: dims={dims[0]:3d}-{dims[-1]:3d} ({len(dims):3d})  '
          f'rms={rms:.6f}  /bias={bias:.1f}  = {scaled:.6f}')

# ── 对比官方 ──
official_mean = np.load('/data/motion/SnapMoGen/meta_data/mean.npy')
official_std  = np.load('/data/motion/SnapMoGen/meta_data/std.npy')

print(f'\n{"":<25s} {"repro_std":>12s} {"official_std":>12s} {"diff":>12s}')
print(f'{"-"*60}')
for name, (dims, _) in groups.items():
    r = repro_std[dims[0]]
    o = official_std[dims[0]]
    print(f'{name:<25s} {r:>12.6f} {o:>12.6f} {abs(r-o):>12.6f}')

mean_diff = np.abs(repro_mean - official_mean).max()
print(f'\nmean max diff: {mean_diff:.6f}')
print(f'std  max diff: {np.abs(repro_std - official_std).max():.6f}')
print(f'std 完全一致: {np.allclose(repro_std, official_std, atol=0.001)}')
print(f'mean 完全一致: {np.allclose(repro_mean, official_mean, atol=0.01)}')

# ── 验证重现的归一化效果 ──
test_files = train_files[::50]
z_vals = []
for f in test_files:
    data = np.load(f)
    z = (data - repro_mean) / np.maximum(repro_std, 1e-8)
    z_vals.append(z)
z_all = np.concatenate(z_vals, 0)
dim_std = z_all.std(0)
in_range = ((dim_std >= 0.8) & (dim_std <= 1.2)).sum()
print(f'\n重现归一化后: std在[0.8,1.2]的维度 = {in_range}/296')
print(f'per-dim std 范围: [{dim_std.min():.4f}, {dim_std.max():.4f}]')
