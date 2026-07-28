"""诊断脚本：校准 SnapMoGen evaluator 上 FID 的量级，对比 fid_emb(index0) 与 VAE mu(index1)。

不修改任何训练代码，只做只读诊断。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import torch
from os.path import join as pjoin

from utils.config_utils import load_config
from models.snapmogen_evaluator import EvaluatorWrapper
from dataset.snapmogen_dataset import TextMotionDataset
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity


def build_loader(batch_size=100):
    cfg = load_config('./SnapMoGen/config/eval_momaskplus.yaml')
    cfg.data.root_dir = '/data/motion/SnapMoGen'
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))
    ds = TextMotionDataset(cfg, mean, std,
                           pjoin(split_dir, 'test_fnames.txt'),
                           pjoin(split_dir, 'test_ids.txt'),
                           pjoin(cfg.data.root_dir, 'all_caption_clean.json'))
    print('test set size =', len(ds))
    return torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=0, drop_last=True)


def build_wrapper():
    cfg = load_config('./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml')
    cfg.data.root_dir = '/data/motion/SnapMoGen'
    cfg.exp.root_ckpt_dir = './SnapMoGen/checkpoint_dir'
    w = EvaluatorWrapper(cfg, device=torch.device('cuda'))
    w.eval()
    return w


@torch.no_grad()
def embed(wrapper, motions, lens, sample_mean=True):
    """返回 (fid_emb, mu_or_sample)"""
    fid_em, vec, _ = wrapper.encode_motion(motions[..., :148].float().cuda(), lens, sample_mean=sample_mean)
    return fid_em.cpu().numpy(), vec.cpu().numpy()


def smooth(m, k=9):
    """时间维度均值滤波，模拟 diffusion 过度平滑的输出"""
    x = m.permute(0, 2, 1)                                   # (b,c,t)
    pad = k // 2
    x = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(x, (pad, pad), mode='replicate'), k, stride=1)
    return x.permute(0, 2, 1)


def main():
    loader = build_loader()
    wrapper = build_wrapper()

    variants = {
        'gt_copy':      lambda m: m.clone(),
        'noise_0.1':    lambda m: m + 0.1 * torch.randn_like(m),
        'noise_0.3':    lambda m: m + 0.3 * torch.randn_like(m),
        'noise_1.0':    lambda m: m + 1.0 * torch.randn_like(m),
        'smooth_k9':    lambda m: smooth(m, 9),
        'smooth_k31':   lambda m: smooth(m, 31),
        'time_shuffle': lambda m: m[:, torch.randperm(m.shape[1])],
        'pure_randn':   lambda m: torch.randn_like(m),
        'all_zero':     lambda m: torch.zeros_like(m),
    }

    acc = {'REF_fid': [], 'REF_mu': []}
    for k in variants:
        acc[k + '_fid'] = []
        acc[k + '_mu'] = []

    for i, batch in enumerate(loader):
        if i >= 10:
            break
        caption, motions, m_lens = batch
        motions = motions.float()
        # 参考组：GT 本身（sample_mean=True，与 activation_dict 一致）
        f, u = embed(wrapper, motions, m_lens, sample_mean=True)
        acc['REF_fid'].append(f); acc['REF_mu'].append(u)
        for name, fn in variants.items():
            f, u = embed(wrapper, fn(motions), m_lens, sample_mean=True)
            acc[name + '_fid'].append(f); acc[name + '_mu'].append(u)
        print(f'batch {i} done')

    for k in acc:
        acc[k] = np.concatenate(acc[k], 0)

    for space in ['fid', 'mu']:
        ref = acc['REF_' + space]
        mu0, cov0 = calculate_activation_statistics(ref)
        print(f'\n===== embedding = {"fid_emb(index0, 官方用法)" if space == "fid" else "VAE mu(index1, 本仓库用法)"} '
              f'dim={ref.shape[1]}  |mu|={np.linalg.norm(mu0):.2f} =====')
        print(f'{"variant":<14}{"FID":>12}{"Diversity":>12}')
        print(f'{"[GT自身]":<14}{0.0:>12.4f}{calculate_diversity(ref, 300):>12.4f}')
        for name in variants:
            e = acc[name + '_' + space]
            m1, c1 = calculate_activation_statistics(e)
            fid = calculate_frechet_distance(mu0, cov0, m1, c1)
            div = calculate_diversity(e, 300)
            print(f'{name:<14}{fid:>12.4f}{div:>12.4f}')


if __name__ == '__main__':
    main()
