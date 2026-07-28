"""诊断3：用"人造去噪器"跑完整 50 步反向链，分别检验
   (a) 采样器/schedule 本身是否健康
   (b) x0 预测误差 -> FID 的映射（需要多准才能到低 FID）
   (c) CFG guidance=2.5 的放大效应会带来多少 FID
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
import diffusion.gaussian_diffusion as gd


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


class Chain:
    """复刻 gaussian_diffusion_simple 的 50 步 DDPM 祖先采样"""
    def __init__(self, steps=50):
        betas = np.array(gd.get_named_beta_schedule('cosine', steps, 1.), dtype=np.float64)
        alphas = 1.0 - betas
        ab = np.cumprod(alphas, 0)
        ab_prev = np.append(1.0, ab[:-1])
        self.T = steps
        self.pv = betas * (1.0 - ab_prev) / (1.0 - ab)
        self.plv = np.log(np.append(self.pv[1], self.pv[1:]))
        self.c1 = betas * np.sqrt(ab_prev) / (1.0 - ab)
        self.c2 = (1.0 - ab_prev) * np.sqrt(alphas) / (1.0 - ab)

    def sample_loop(self, shape, denoiser, device='cuda'):
        xt = torch.randn(shape, device=device)
        for i in range(self.T - 1, -1, -1):
            pred_x0 = denoiser(xt, i)
            mean = self.c1[i] * pred_x0 + self.c2[i] * xt
            if i != 0:
                mean = mean + np.exp(0.5 * self.plv[i]) * torch.randn_like(xt)
            xt = mean
        return xt


def main():
    loader = build_loader()
    wrapper = build_wrapper()
    chain = Chain(50)

    CFG = 2.5
    configs = {
        'oracle(完美预测)':      dict(err=0.0,  cfg=1.0),
        'oracle+err0.1':         dict(err=0.1,  cfg=1.0),
        'oracle+err0.2':         dict(err=0.2,  cfg=1.0),
        'oracle+err0.3':         dict(err=0.3,  cfg=1.0),
        'oracle+CFG2.5':         dict(err=0.0,  cfg=CFG),
        'oracle+err0.1+CFG2.5':  dict(err=0.1,  cfg=CFG),
        'oracle+err0.2+CFG2.5':  dict(err=0.2,  cfg=CFG),
    }
    acc = {'REF': []}
    for k in configs:
        acc[k] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 6:
                break
            _, motions, m_lens = batch
            x0 = motions.float().cuda()
            _, v, _ = wrapper.encode_motion(x0[..., :148], m_lens, sample_mean=True)
            acc['REF'].append(v.cpu().numpy())

            # 无条件分支的"泛化预测"：batch 内的平均动作（与文本无关）
            uncond_target = x0.mean(dim=0, keepdim=True).expand_as(x0)

            for name, c in configs.items():
                err, s = c['err'], c['cfg']

                def denoiser(xt, i, err=err, s=s):
                    cond = x0 + err * torch.randn_like(x0)
                    if s == 1.0:
                        return cond
                    unc = uncond_target + err * torch.randn_like(x0)
                    return unc + s * (cond - unc)

                out = chain.sample_loop(x0.shape, denoiser)
                _, v, _ = wrapper.encode_motion(out[..., :148], m_lens, sample_mean=True)
                acc[name].append(v.cpu().numpy())
            print(f'batch {bi} done')

    for k in acc:
        acc[k] = np.concatenate(acc[k], 0)
    mu0, cov0 = calculate_activation_statistics(acc['REF'])

    print(f'\n===== 50步反向链 + 人造去噪器 (GT Diversity={calculate_diversity(acc["REF"], 300):.2f}) =====')
    print(f'{"配置":<24}{"FID":>12}{"Diversity":>12}')
    for name in configs:
        e = acc[name]
        m1, c1 = calculate_activation_statistics(e)
        print(f'{name:<24}{calculate_frechet_distance(mu0, cov0, m1, c1):>12.2f}{calculate_diversity(e, 300):>12.2f}')


if __name__ == '__main__':
    main()
