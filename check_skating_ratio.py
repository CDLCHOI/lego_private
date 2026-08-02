"""
check_skating_ratio.py
简单脚本：加载3个模型，对测试集生成 motion，计算脚滑比例（Skating Ratio）。
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

MODELS = OrderedDict({
    'MDM': {
        'ckpt': 'output/0814_MDMCLIP_b128/net_best.pth',
        'add_clip_lora': False,
    },
    'MDM+LeGO-CLIP': {
        'ckpt': 'output/0911_MDMCLIP_preatrainlora_ric1_b64/net_best.pth',
        'add_clip_lora': True,
    },
    'LeGO': {
        'ckpt': 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/net_best.pth',
        'add_clip_lora': True,
    },
})

GPU = '6'
BATCH_SIZE = 32
MAX_SAMPLES = 1000  # 跟日志（10000）不同，但脚滑比例在1000样本上已经非常接近


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ['OMP_NUM_THREADS'] = '8'

import options.option_transformer as option_trans
args = option_trans.get_args_parser()

from data_loaders.humanml.utils.metrics import calculate_skating_ratio, calculate_skating_ratio_kit
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
from dataset import dataset_control
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from utils.model_util import create_gaussian_diffusion_simple, get_mdm_bert_args
from utils.lora_util import load_lora_mdm_for_eval
from utils.mask_utils import load_ckpt
from utils.motion_process import recover_from_ric
from models.mdm_bert.mdm_bert import MDMBERT


def load_model(ckpt_path, add_clip_lora):
    """加载模型，完全复用 test_direction_speed.py::build_model() 的逻辑"""
    args.add_clip_lora = add_clip_lora
    net = MDMBERT(**get_mdm_bert_args(args, 'mdm_bert'))
    if add_clip_lora:
        load_lora_mdm_for_eval(net, ckpt_path)
    else:
        load_ckpt(net, ckpt_path, key=None, strict=False)
    diffusion = create_gaussian_diffusion_simple(args, net, 'mdm_bert')
    net.cuda()
    net.eval()
    return net, diffusion


def main():
    # 设置必要的 args（必须在 build model 之前！）
    args.dataset_name = 't2m'
    args.modeltype = 'mdm_bert'
    args.text_encoder_type = 'clip'
    args.batch_size = BATCH_SIZE
    args.max_samples = MAX_SAMPLES
    args.eval_mode = 'no_mm'
    args.normalize_traj = True
    args.diffusion_steps = 50
    args.guidance_param = 2.5

    # 数据加载器
    gen_loader = dataset_control.DataLoader(
        batch_size=args.batch_size, args=args, mode='eval',
        split='test', shuffle=False, num_workers=0, drop_last=True,
    )

    print('=' * 60)
    print('  模型脚滑比例 (Skating Ratio) 对比')
    print(f'  样本数: {MAX_SAMPLES}')
    print('=' * 60)

    for model_name, cfg in MODELS.items():
        print(f'\n>>> 加载模型: {model_name} ...')
        net, diffusion = load_model(cfg['ckpt'], cfg['add_clip_lora'])

        # 生成 motion
        print(f'    生成 motion 中...')
        motion_loader, _ = get_control_dataset(
            args, gen_loader, None, None, diffusion,
            mm_num_samples=0, mm_num_repeats=0, num_samples_limit=MAX_SAMPLES,
        )

        # 计算脚滑比例
        skate_ratio_sum = 0
        all_size = 0
        with torch.no_grad():
            for batch in motion_loader:
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, hint, filename = batch
                dim = motions.shape[-1]
                mean_for_eval = motion_loader.dataset.gen_loader.dataset.mean_for_eval[:dim]
                std_for_eval = motion_loader.dataset.gen_loader.dataset.std_for_eval[:dim]
                motions = motions * std_for_eval + mean_for_eval
                motions = motions.float()
                n_joints = 22 if motions.shape[-1] in [263, 67] else 21
                joints = recover_from_ric(motions, n_joints)
                if n_joints == 21:
                    skate_ratio, _ = calculate_skating_ratio_kit(joints.permute(0, 2, 3, 1))
                else:
                    skate_ratio, _ = calculate_skating_ratio(joints.permute(0, 2, 3, 1))
                skate_ratio_sum += skate_ratio.sum().item()
                all_size += joints.shape[0]

        skating_ratio = skate_ratio_sum / all_size
        print(f'    [{model_name}] Skating Ratio = {skating_ratio:.4f}')

        # 清理 GPU 内存
        del net, diffusion, motion_loader
        torch.cuda.empty_cache()

    print('\n' + '=' * 60)
    print('  日志参考值:')
    print('    MDM              : 0.0963 (10 replications × 10000 samples)')
    print('    MDM+LeGO-CLIP    : 0.0991 (10 replications × 10000 samples)')
    print('    LeGO             : 0.0612 (10 replications × 10000 samples)')
    print('=' * 60)


if __name__ == '__main__':
    main()
