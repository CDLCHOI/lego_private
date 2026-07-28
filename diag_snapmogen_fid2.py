"""诊断2：把 FID 数值翻译成"等效加噪步数"，并检查 296 维特征的逐维统计。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import torch
from os.path import join as pjoin

from utils.config_utils import load_config
from models.snapmogen_evaluator import EvaluatorWrapper
from dataset.snapmogen_dataset import TextMotionDataset
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity
import diffusion.gaussian_diffusion as gd

MEAN_PATH = '/data/motion/SnapMoGen/meta_data/mean.npy'
STD_PATH = '/data/motion/SnapMoGen/meta_data/std.npy'


def per_dim_stats():
    """全训练集的逐维归一化后统计"""
    import glob
    m = np.load(MEAN_PATH); s = np.load(STD_PATH)
    fs = sorted(glob.glob('/data/motion/SnapMoGen/renamed_feats/*.npy'))
    fs = fs[::20]
    X = np.concatenate([np.load(f) for f in fs], 0)
    Z = (X - m) / s
    print(f'\n===== 逐维统计 (共{len(fs)}个文件, {X.shape[0]}帧) =====')
    print(f'归一化后 逐维mean: min={Z.mean(0).min():.3f} max={Z.mean(0).max():.3f}')
    print(f'归一化后 逐维std : min={Z.std(0).min():.4f} max={Z.std(0).max():.3f}')
    bad_std = np.where(Z.std(0) < 0.05)[0]
    print(f'std<0.05 的维度: {bad_std}')
    big = np.where(np.abs(Z).max(0) > 50)[0]
    print(f'|Z|>50 的维度: {big}  (其中在前148维的: {big[big < 148]})')
    print(f'前148维 |Z|max={np.abs(Z[:, :148]).max():.1f}, 后148维 |Z|max={np.abs(Z[:, 148:]).max():.1f}')
    return Z.std(0)


def build_loader(batch_size=100):
    cfg = load_config('./SnapMoGen/config/eval_momaskplus.yaml')
    cfg.data.root_dir = '/data/motion/SnapMoGen'
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    ds = TextMotionDataset(cfg, np.load(pjoin(meta_dir, 'mean.npy')), np.load(pjoin(meta_dir, 'std.npy')),
                           pjoin(split_dir, 'test_fnames.txt'), pjoin(split_dir, 'test_ids.txt'),
                           pjoin(cfg.data.root_dir, 'all_caption_clean.json'))
    return torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=0, drop_last=True)


def build_wrapper():
    cfg = load_config('./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml')
    cfg.exp.root_ckpt_dir = './SnapMoGen/checkpoint_dir'
    w = EvaluatorWrapper(cfg, device=torch.device('cuda'))
    w.eval()
    return w


def main():
    per_dim_stats()

    # 50 步 cosine schedule 的 alpha_bar
    betas = gd.get_named_beta_schedule('cosine', 50, 1.)
    abar = np.cumprod(1.0 - betas)
    sqrt_ab = np.sqrt(abar)
    sqrt_1mab = np.sqrt(1 - abar)
    print('\n===== 50步 cosine schedule =====')
    for t in [0, 1, 2, 5, 10, 15, 20, 30, 49]:
        print(f'  t={t:2d}  alpha_bar={abar[t]:.5f}  信号系数={sqrt_ab[t]:.4f}  噪声系数={sqrt_1mab[t]:.4f}')

    loader = build_loader()
    wrapper = build_wrapper()

    ts = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    acc = {'REF': []}
    for t in ts:
        acc[f't{t}'] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            _, motions, m_lens = batch
            motions = motions.float().cuda()
            _, v, _ = wrapper.encode_motion(motions[..., :148], m_lens, sample_mean=True)
            acc['REF'].append(v.cpu().numpy())
            for t in ts:
                noisy = sqrt_ab[t] * motions + sqrt_1mab[t] * torch.randn_like(motions)
                _, v, _ = wrapper.encode_motion(noisy[..., :148], m_lens, sample_mean=True)
                acc[f't{t}'].append(v.cpu().numpy())

    for k in acc:
        acc[k] = np.concatenate(acc[k], 0)
    mu0, cov0 = calculate_activation_statistics(acc['REF'])

    print('\n===== FID(GT, q_sample(GT, t))  —— 把 FID 翻译成等效加噪步数 =====')
    print(f'{"t":>4}{"FID":>12}{"Diversity":>12}')
    for t in ts:
        e = acc[f't{t}']
        m1, c1 = calculate_activation_statistics(e)
        print(f'{t:>4}{calculate_frechet_distance(mu0, cov0, m1, c1):>12.2f}{calculate_diversity(e, 300):>12.2f}')


if __name__ == '__main__':
    main()
