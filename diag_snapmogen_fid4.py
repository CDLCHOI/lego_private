"""诊断4：CFG guidance 强度扫描 + 更真实的"结构化误差"去噪器。"""
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
from diag_snapmogen_fid3 import build_loader, build_wrapper, Chain


def smooth(m, k):
    x = m.permute(0, 2, 1)
    pad = k // 2
    x = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(x, (pad, pad), mode='replicate'), k, stride=1)
    return x.permute(0, 2, 1)


def main():
    loader = build_loader()
    wrapper = build_wrapper()
    chain = Chain(50)

    # uncond 分支的两种建模：'mean' = batch 平均动作（温和）；'shift' = 另一条真实动作（激进，更接近实际）
    scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    configs = {}
    for s in scales:
        configs[f'uncond=mean  s={s}'] = ('mean', s, 1)
        configs[f'uncond=其他动作 s={s}'] = ('shift', s, 1)
    # 结构化误差（过度平滑）+ 不同 guidance
    for s in [1.0, 2.5]:
        configs[f'平滑k9 uncond=mean s={s}'] = ('mean', s, 9)

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

            mean_motion = x0.mean(dim=0, keepdim=True).expand_as(x0)
            other_motion = x0.roll(shifts=1, dims=0)

            for name, (utype, s, k) in configs.items():
                cond_pred = x0 if k == 1 else smooth(x0, k)
                unc_pred = mean_motion if utype == 'mean' else other_motion

                def denoiser(xt, i, s=s, c=cond_pred, u=unc_pred):
                    return c if s == 1.0 else u + s * (c - u)

                out = chain.sample_loop(x0.shape, denoiser)
                _, v, _ = wrapper.encode_motion(out[..., :148], m_lens, sample_mean=True)
                acc[name].append(v.cpu().numpy())
            print(f'batch {bi} done')

    for k in acc:
        acc[k] = np.concatenate(acc[k], 0)
    mu0, cov0 = calculate_activation_statistics(acc['REF'])

    print(f'\n===== guidance 扫描 (GT Diversity={calculate_diversity(acc["REF"], 300):.2f}) =====')
    print(f'{"配置":<28}{"FID":>12}{"Diversity":>12}')
    for name in configs:
        e = acc[name]
        m1, c1 = calculate_activation_statistics(e)
        print(f'{name:<28}{calculate_frechet_distance(mu0, cov0, m1, c1):>12.2f}{calculate_diversity(e, 300):>12.2f}')


if __name__ == '__main__':
    main()
