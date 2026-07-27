"""
测试 CLIP 和 LEGO-CLIP 对 left/right 方向词汇的敏感度。

核心思路:
    - 取一条包含 left/right 的文本描述（原文本），对应一个原始 motion
    - 将文本中的 left↔right 互换，得到镜像文本
    - 用同一个原始 motion，分别计算:
        * 原文本 vs 原motion 的相似度
        * 镜像文本 vs 原motion 的相似度
    - 期望: LEGO-CLIP 的"原文本-原motion相似度"明显高于"镜像文本-原motion相似度"
            而 CLIP 的两者应该几乎相等

用法:
    python test_legoclip_motion_similarity.py
    python test_legoclip_motion_similarity.py --max_samples 300 --verbose
    python test_legoclip_motion_similarity.py --batch_size 64 --output results.csv
"""

import os
import re
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm


# ───────────────────────────── 工具函数 ─────────────────────────────


def build_dummy_args():
    """创建 EvaluatorMDMWrapper 所需的虚拟 args"""
    class DummyArgs:
        pass
    return DummyArgs()


def mirror_text(text: str) -> str:
    """
    将文本中的 left↔right 互换，保持大小写。

    Examples:
        "turn left"  → "turn right"
        "Turn Left"  → "Turn Right"
        "TURN LEFT"  → "TURN RIGHT"
    """
    def swap(match):
        word = match.group(0)
        lower = word.lower()
        if lower == 'left':
            if word.isupper():
                return 'RIGHT'
            elif word[0].isupper():
                return 'Right'
            else:
                return 'right'
        elif lower == 'right':
            if word.isupper():
                return 'LEFT'
            elif word[0].isupper():
                return 'Left'
            else:
                return 'left'
        return word  # 不会发生

    return re.sub(r'\bleft\b|\bright\b', swap, text, flags=re.IGNORECASE)


def extract_text_from_line(line: str):
    """
    从一行 'text#word1/POS#word2/POS#...#start#end' 中提取纯文本描述。
    """
    line = line.strip()
    if not line:
        return None, None

    parts = line.split('#')
    if len(parts) < 3:
        return None, None

    text = parts[0]
    start_end = (parts[-2], parts[-1])
    return text, start_end


def pick_text_line(filepath: str):
    """
    从文本文件中选取一行:
    - 从第1行开始，找到以 '#0.0#0.0' 结尾的行
    - 如果第1行不是以此结尾，继续往下找
    """
    if not os.path.isfile(filepath):
        return None, None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith('#0.0#0.0'):
            text, _ = extract_text_from_line(line)
            if text:
                return text, line
    return None, None


def collate_motions(motion_list):
    """
    将一个 list of (T, D) numpy arrays 打包为 (B, T_max, D) tensor + lengths tensor。
    """
    lengths = [m.shape[0] for m in motion_list]
    max_len = max(lengths)
    D = motion_list[0].shape[-1]
    batch = torch.zeros(len(motion_list), max_len, D)
    for i, m in enumerate(motion_list):
        batch[i, :m.shape[0]] = torch.from_numpy(m).float()
    return batch, torch.tensor(lengths)


# ───────────────────────────── 主逻辑 ─────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='测试 CLIP / LEGO-CLIP 对 left/right 方向词汇的敏感度'
    )
    parser.add_argument('--data_dir', type=str,
                        default='dataset/HumanML3D/new_joint_vecs')
    parser.add_argument('--text_dir', type=str,
                        default='dataset/HumanML3D/texts')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar')
    parser.add_argument('--clip_version', type=str, default='ViT-B/32')
    parser.add_argument('--adaclip_ckpt', type=str,
                        default='output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/merged_clip.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('  CLIP vs LEGO-CLIP: left/right 方向敏感度测试')
    print('=' * 65)
    print(f'  设备:              {device}')
    print(f'  数据目录:          {args.data_dir}')
    print(f'  文本目录:          {args.text_dir}')
    print(f'  Motion ckpt:       {args.checkpoint}')
    print(f'  AdaCLIP ckpt:      {args.adaclip_ckpt}')
    print(f'  Batch size:        {args.batch_size}')
    print(f'  CLIP 版本:         {args.clip_version}')
    print()
    print('  测试方法: 原文本(含left/right) vs 原motion')
    print('            镜像文本(left↔right互换) vs 同一个原motion')
    print('  期望结果: LEGO-CLIP 对方向变化更敏感（原>镜像），CLIP 两者相近')

    # ──── 阶段1: 收集符合条件的样本 ────
    print('\n[阶段1] 收集符合条件的样本...')

    all_files = sorted([
        f for f in os.listdir(args.data_dir)
        if f.endswith('.npy') and not f.startswith('M')
    ])

    samples = []  # {'filename', 'orig_text', 'mirr_text', 'orig_path'}

    for filename in tqdm(all_files, desc='筛选样本'):
        text_path = os.path.join(args.text_dir, filename.replace('.npy', '.txt'))
        orig_text, _ = pick_text_line(text_path)

        if orig_text is None:
            continue

        # 检查 left / right
        text_lower = orig_text.lower()
        if 'left' not in text_lower and 'right' not in text_lower:
            continue

        # 生成镜像文本
        mirr_text = mirror_text(orig_text)

        # 确认确实发生了替换
        if mirr_text == orig_text:
            continue

        orig_path = os.path.join(args.data_dir, filename)

        samples.append({
            'filename': filename,
            'orig_text': orig_text,
            'mirr_text': mirr_text,
            'orig_path': orig_path,
        })

        if args.max_samples and len(samples) >= args.max_samples:
            break

    num_samples = len(samples)
    print(f'  符合条件的样本数: {num_samples}')

    if num_samples == 0:
        print('  没有符合条件的样本，退出。')
        return

    # 打印几个镜像文本示例
    print('\n  镜像文本示例（前3个）:')
    for s in samples[:3]:
        print(f'    [{s["filename"]}]')
        print(f'      原文本: {s["orig_text"]}')
        print(f'      镜像文本: {s["mirr_text"]}')

    # ──── 阶段2: 加载 Motion Encoder ────
    print('\n[阶段2] 加载 Motion Encoder (EvaluatorMDMWrapper)...')
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

    motion_encoder = EvaluatorMDMWrapper(
        't2m', device, build_dummy_args(), ckpt_path=args.checkpoint
    )
    print('  Motion Encoder 加载完成!')

    # ──── 阶段3: 加载两种 Text Encoder ────
    print('\n[阶段3] 加载文本编码器...')

    print('  加载原始 CLIP...')
    clip_model, _ = clip.load(args.clip_version, device='cpu', jit=False)
    clip_model = clip_model.float().to(device)
    clip_model.eval()

    print('  加载 AdaCLIP (LEGO-CLIP)...')
    adaclip_model, _ = clip.load(args.clip_version, device='cpu', jit=False)
    merge_state = torch.load(args.adaclip_ckpt, map_location='cpu')
    adaclip_model.load_state_dict(merge_state, strict=True)
    adaclip_model = adaclip_model.float().to(device)
    adaclip_model.eval()

    print('  两种文本编码器加载完成!')

    # ──── 阶段4: 批量计算相似度 ────
    print(f'\n[阶段4] 批量计算相似度 (batch_size={args.batch_size})...')

    # 存储结果: {model: {'orig': [...], 'mirr': [...]}}
    # 'orig' = 原文本 vs 原motion
    # 'mirr' = 镜像文本 vs 原motion
    all_results = {
        'CLIP':       {'orig': [], 'mirr': []},
        'LEGO-CLIP':  {'orig': [], 'mirr': []},
    }

    detail_records = []

    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    pbar = tqdm(total=num_samples, desc='计算相似度')

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, num_samples)
        batch_samples = samples[start:end]
        batch_orig_texts = [s['orig_text'] for s in batch_samples]
        batch_mirr_texts = [s['mirr_text'] for s in batch_samples]

        # ── 4a. 加载并编码 motions（只用原motion） ──
        orig_motions_data = [np.load(s['orig_path']) for s in batch_samples]
        orig_batch, orig_lens = collate_motions(orig_motions_data)

        with torch.no_grad():
            orig_emb = motion_encoder.get_motion_embeddings(orig_batch, orig_lens)
        orig_emb = F.normalize(orig_emb, dim=1)  # (B, 512)

        # ── 4b. 编码原文本和镜像文本 ──
        with torch.no_grad():
            tokenized_orig = clip.tokenize(batch_orig_texts, truncate=True).to(device)
            tokenized_mirr = clip.tokenize(batch_mirr_texts, truncate=True).to(device)

            # CLIP
            clip_orig_text_emb = clip_model.encode_text(tokenized_orig).float()
            clip_mirr_text_emb = clip_model.encode_text(tokenized_mirr).float()
            # LEGO-CLIP
            adaclip_orig_text_emb = adaclip_model.encode_text(tokenized_orig).float()
            adaclip_mirr_text_emb = adaclip_model.encode_text(tokenized_mirr).float()

        clip_orig_text_emb = F.normalize(clip_orig_text_emb, dim=1)
        clip_mirr_text_emb = F.normalize(clip_mirr_text_emb, dim=1)
        adaclip_orig_text_emb = F.normalize(adaclip_orig_text_emb, dim=1)
        adaclip_mirr_text_emb = F.normalize(adaclip_mirr_text_emb, dim=1)

        # ── 4c. 计算余弦相似度 ──
        # 核心比较:
        #   CLIP:      原文本+原motion vs 镜像文本+原motion
        #   LEGO-CLIP:  原文本+原motion vs 镜像文本+原motion
        clip_orig_sim   = (clip_orig_text_emb * orig_emb).sum(dim=1)
        clip_mirr_sim   = (clip_mirr_text_emb * orig_emb).sum(dim=1)
        adaclip_orig_sim = (adaclip_orig_text_emb * orig_emb).sum(dim=1)
        adaclip_mirr_sim = (adaclip_mirr_text_emb * orig_emb).sum(dim=1)

        # 收集结果
        for j in range(len(batch_samples)):
            all_results['CLIP']['orig'].append(clip_orig_sim[j].item())
            all_results['CLIP']['mirr'].append(clip_mirr_sim[j].item())
            all_results['LEGO-CLIP']['orig'].append(adaclip_orig_sim[j].item())
            all_results['LEGO-CLIP']['mirr'].append(adaclip_mirr_sim[j].item())

            if args.verbose:
                s = batch_samples[j]

                print(
                    f'{s["filename"]}: '
                    f'CLIP(原文本={clip_orig_sim[j].item():.4f}, 镜像文本={clip_mirr_sim[j].item():.4f}), '
                    f'LEGO(原文本={adaclip_orig_sim[j].item():.4f}, 镜像文本={adaclip_mirr_sim[j].item():.4f})'
                )
                print(f'    原文本: {s["orig_text"]}')

            detail_records.append({
                'filename': batch_samples[j]['filename'],
                'orig_text': batch_samples[j]['orig_text'],
                'mirr_text': batch_samples[j]['mirr_text'],
                'CLIP_orig': clip_orig_sim[j].item(),
                'CLIP_mirr': clip_mirr_sim[j].item(),
                'LEGO_orig': adaclip_orig_sim[j].item(),
                'LEGO_mirr': adaclip_mirr_sim[j].item(),
            })

        pbar.update(len(batch_samples))

    pbar.close()

    # ──── 阶段5: 统计与输出 ────
    print('\n' + '=' * 65)
    print('                    统 计 结 果')
    print('=' * 65)
    print(f'  样本总数: {num_samples}')
    print()
    print('  比较: 原文本 vs 原motion | 镜像文本(left↔right互换) vs 原motion')
    print()

    header = (f'{"模型":<14} {"原文本+原Motion":>18} {"镜像文本+原Motion":>18} '
              f'{"差值(原-镜像)":>18}')
    print(header)
    print('-' * len(header))

    for model_name in ['CLIP', 'LEGO-CLIP']:
        orig_arr = np.array(all_results[model_name]['orig'])
        mirr_arr = np.array(all_results[model_name]['mirr'])

        orig_mean = orig_arr.mean()
        mirr_mean = mirr_arr.mean()
        diff_mean = orig_mean - mirr_mean  # 正值 = 原文本更相似 ✓

        print(f'{model_name:<14} {orig_mean:>18.6f} {mirr_mean:>18.6f} {diff_mean:>18.6f}')

    # 关键指标: 原文本 > 镜像文本 的比例
    print('\n' + '-' * 65)
    print('  ★ 关键指标: 原文本相似度 > 镜像文本相似度的样本比例')
    print('    (比例越高 = 对方向变换越敏感)')
    for model_name in ['CLIP', 'LEGO-CLIP']:
        orig_arr = np.array(all_results[model_name]['orig'])
        mirr_arr = np.array(all_results[model_name]['mirr'])
        pct = (orig_arr > mirr_arr).mean() * 100
        avg_diff = (orig_arr - mirr_arr).mean()
        print(f'    {model_name:<14}: {pct:.2f}%  '
              f'({int((orig_arr > mirr_arr).sum())}/{len(orig_arr)}), '
              f'平均差异={avg_diff:.6f}')

    # 详细统计
    print('\n' + '-' * 65)
    print('  详细统计:')
    print(f'{"模型":<14} {"文本类型":<10} {"均值":>10} {"标准差":>10} '
          f'{"中位数":>10} {"最小值":>10} {"最大值":>10}')
    print('-' * 80)

    for model_name in ['CLIP', 'LEGO-CLIP']:
        for label, key in [('原文本', 'orig'), ('镜像文本', 'mirr')]:
            arr = np.array(all_results[model_name][key])
            print(f'{model_name:<14} {label:<10} {arr.mean():>10.6f} {arr.std():>10.6f} '
                  f'{np.median(arr):>10.6f} {arr.min():>10.6f} {arr.max():>10.6f}')

    # 分布区间统计
    print('\n' + '-' * 65)
    print('  原文本-镜像文本 相似度差值分布（正值=原文本更相似）:')
    bins = [(-1, 0), (0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1)]
    for model_name in ['CLIP', 'LEGO-CLIP']:
        orig_arr = np.array(all_results[model_name]['orig'])
        mirr_arr = np.array(all_results[model_name]['mirr'])
        diff = orig_arr - mirr_arr
        print(f'    {model_name}:')
        for lo, hi in bins:
            count = int(((diff > lo) & (diff <= hi)).sum())
            pct = count / len(diff) * 100
            bar = '█' * int(pct / 2)
            label = f'< 0' if lo == -1 else f'{lo:.2f}~{hi:.2f}' if hi != 1 else f'> {lo:.2f}'
            print(f'      diff ∈ [{label}]: {count:>6d} ({pct:>5.1f}%) {bar}')

    # 展示一些典型案例
    print('\n' + '-' * 65)
    print('  典型案例（LEGO-CLIP 原文本 > 镜像文本，差异最大的前 10 个）:')
    sorted_by_diff = sorted(
        detail_records,
        key=lambda r: r['LEGO_orig'] - r['LEGO_mirr'],
        reverse=True
    )
    for r in sorted_by_diff[:10]:
        diff_lego = r['LEGO_orig'] - r['LEGO_mirr']
        diff_clip = r['CLIP_orig'] - r['CLIP_mirr']
        print(f'    [{r["filename"]}] LEGO_diff={diff_lego:.4f} CLIP_diff={diff_clip:.4f}')
        print(f'      原文本: {r["orig_text"]}')
        print(f'      镜像文本: {r["mirr_text"]}')

    print('\n' + '=' * 65)

    # ──── 保存 CSV ────
    output_path = args.output or 'test_legoclip_similarity_results.csv'
    with open(output_path, 'w') as f:
        f.write('filename,orig_text,mirr_text,CLIP_orig,CLIP_mirr,LEGO_orig,LEGO_mirr\n')
        for r in detail_records:
            escaped_orig = r['orig_text'].replace('"', '""')
            escaped_mirr = r['mirr_text'].replace('"', '""')
            f.write(f'{r["filename"]},"{escaped_orig}","{escaped_mirr}",'
                    f'{r["CLIP_orig"]:.6f},{r["CLIP_mirr"]:.6f},'
                    f'{r["LEGO_orig"]:.6f},{r["LEGO_mirr"]:.6f}\n')
    print(f'详细结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
