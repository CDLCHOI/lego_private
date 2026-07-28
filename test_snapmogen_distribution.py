"""对比官方 mean/std 与新计算的 mean/std 在 SnapMoGen 全数据集上的归一化效果"""
import numpy as np, glob, os

feat_dir = '/data/motion/SnapMoGen/renamed_feats'
old_mean = np.load('/data/motion/SnapMoGen/meta_data/mean.npy')
old_std  = np.load('/data/motion/SnapMoGen/meta_data/std.npy')
new_mean = np.load('/home/deli/project/reward_mdm/dataset/snapmogen_norm/mean.npy')
new_std  = np.load('/home/deli/project/reward_mdm/dataset/snapmogen_norm/std.npy')

# 采样（每10个取1个，覆盖全数据集, 约100万帧）
files = sorted(glob.glob(os.path.join(feat_dir, '*.npy')))[::10]
print(f'采样 {len(files)} 个文件，加载中...')

old_z = []; new_z = []
for f in files:
    data = np.load(f)
    old_z.append((data - old_mean) / np.maximum(old_std, 1e-8))
    new_z.append((data - new_mean) / np.maximum(new_std, 1e-8))
old_z = np.concatenate(old_z, 0)
new_z = np.concatenate(new_z, 0)
print(f'总帧数: {old_z.shape[0]}, 维度: {old_z.shape[1]}')

# ── 逐维统计 ──
old_dim_mean = old_z.mean(0); new_dim_mean = new_z.mean(0)
old_dim_std  = old_z.std(0);  new_dim_std  = new_z.std(0)
old_dim_max  = np.abs(old_z).max(0); new_dim_max = np.abs(new_z).max(0)

print(f'\n{"指标":<24} {"官方 mean/std":>18} {"新 mean/std":>18}')
print(f'{"per-dim mean 范围":<24} [{old_dim_mean.min():.4f}, {old_dim_mean.max():.4f}]   [{new_dim_mean.min():.4f}, {new_dim_mean.max():.4f}]')
print(f'{"per-dim std  范围":<24} [{old_dim_std.min():.4f}, {old_dim_std.max():.4f}]   [{new_dim_std.min():.4f}, {new_dim_std.max():.4f}]')
print(f'{"per-dim |z|max 范围":<24} [{old_dim_max.min():.1f}, {old_dim_max.max():.1f}]   [{new_dim_max.min():.1f}, {new_dim_max.max():.1f}]')

# ── 整体分位数 ──
print(f'\n{"分位":<10} {"官方 mean/std":>12} {"新 mean/std":>12}')
for pct in [50, 90, 95, 99, 99.9, 100]:
    print(f'{pct:5.1f}%    {np.percentile(np.abs(old_z),pct):>12.2f} {np.percentile(np.abs(new_z),pct):>12.2f}')

# ── 逐维 std 分布 ──
print(f'\n=== 逐维 std 分布（偏离1.0的程度）===')
for label, z in [('官方', old_z), ('新  ', new_z)]:
    dim_std = z.std(0)
    bins = [(0,0.1),(0.1,0.5),(0.5,0.8),(0.8,1.2),(1.2,2.0),(2.0,5.0),(5.0,100)]
    parts = []
    for lo,hi in bins:
        n = ((dim_std >= lo) & (dim_std < hi)).sum()
        parts.append(f'[{lo:.1f},{hi:.1f}):{n}')
    print(f'{label}: {", ".join(parts)}')

# ── 最差维度 ──
print(f'\n=== 官方 归一化 std 最差的10个维度 ===')
worst = np.argsort(np.abs(old_dim_std - 1.0))[-10:][::-1]
for d in worst:
    print(f'  dim{d:3d}: std={old_dim_std[d]:.4f} (偏离 {abs(old_dim_std[d]-1):.4f})  |z|max={old_dim_max[d]:.1f}')

print(f'\n=== 新   归一化 std 最差的10个维度 ===')
worst = np.argsort(np.abs(new_dim_std - 1.0))[-10:][::-1]
for d in worst:
    print(f'  dim{d:3d}: std={new_dim_std[d]:.4f} (偏离 {abs(new_dim_std[d]-1):.4f})  |z|max={new_dim_max[d]:.1f}')

# ── 结论 ──
old_good = ((old_dim_std >= 0.8) & (old_dim_std <= 1.2)).sum()
new_good = ((new_dim_std >= 0.8) & (new_dim_std <= 1.2)).sum()
print(f'\n=== 总结 ===')
print(f'官方: std在[0.8,1.2]的维度 = {old_good}/296, 偏离>0.3的维度 = {(np.abs(old_dim_std-1.0)>0.3).sum()}')
print(f'新  : std在[0.8,1.2]的维度 = {new_good}/296, 偏离>0.3的维度 = {(np.abs(new_dim_std-1.0)>0.3).sum()}')
