"""
eval_mdm_lego_histogram.py
───────────────────────────
Part 1 独立脚本：统计 HumanML3D 测试集所有文本 prompt 的单词数量分布，
绘制直方图，并打印每个文本长度的样本数量和占比。
"""

import os
import codecs as cs
import numpy as np
from os.path import join as pjoin
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 路径常量
# ═══════════════════════════════════════════════════════════════

TEXT_DIR = './dataset/HumanML3D/texts'
TEST_FILE = './dataset/HumanML3D/test.txt'
OUTPUT_DIR = 'output/eval_by_textlength'


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 读取测试集文件列表 ──
    with open(TEST_FILE, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    print(f'测试集 motion 文件总数: {len(test_ids)}')

    # ── 逐个读取文本第一行，统计单词数 ──
    id_to_length = {}
    lengths = []

    for mid in test_ids:
        txt_path = pjoin(TEXT_DIR, f'{mid}.txt')
        try:
            with cs.open(txt_path, 'r') as f:
                first_line = f.readline().strip()
                caption = first_line.split('#')[0]
                word_count = len(caption.split())
                id_to_length[mid] = word_count
                lengths.append(word_count)
        except Exception as e:
            print(f'  [警告] 读取 {txt_path} 失败: {e}')
            continue

    lengths = np.array(lengths)

    # ── 基本统计 ──
    print(f'\n{"=" * 55}')
    print(f'  有效文本数: {len(lengths)}')
    print(f'  单词数范围: {lengths.min()} – {lengths.max()}')
    print(f'  均值: {lengths.mean():.2f}')
    print(f'  中位数: {np.median(lengths):.1f}')
    print(f'  标准差: {lengths.std():.2f}')

    # ── 分位数 ──
    percentiles = [5, 10, 20, 25, 33, 50, 67, 75, 80, 90, 95]
    print(f'\n{"=" * 55}')
    print(f'  分位数统计:')
    for p in percentiles:
        val = np.percentile(lengths, p)
        print(f'    {p:3d}%: {val:6.1f} words')

    # ── 每个文本长度的详细统计 ──
    counter = Counter(lengths)
    print(f'\n{"=" * 55}')
    print(f'  每个单词数的样本数量及占比（按单词数升序）:')
    print(f'  {"单词数":>6s}  {"样本数":>6s}  {"占比":>8s}  {"累积占比":>8s}')
    print(f'  {"-" * 35}')

    cumulative = 0
    total = len(lengths)
    for wc in sorted(counter.keys()):
        count = counter[wc]
        cumulative += count
        pct = count / total * 100
        cum_pct = cumulative / total * 100
        print(f'  {wc:6d}  {count:6d}  {pct:7.2f}%  {cum_pct:7.2f}%')

    # ── 33% / 67% 分位数边界 ──
    p33 = np.percentile(lengths, 33)
    p50 = np.median(lengths)
    p67 = np.percentile(lengths, 67)

    short_upper = int(np.floor(p33))
    medium_lower = short_upper + 1
    medium_upper = int(np.ceil(p67)) - 1
    long_lower = int(np.ceil(p67))

    # 防止边界重叠
    if short_upper >= long_lower:
        short_upper = int(p50)
        long_lower = short_upper + 1
        medium_lower = short_upper + 1
        medium_upper = short_upper
        print(f'\n  ⚠ 33%/67% 分位数边界重叠，回退到中位数 {int(p50)} 划分')

    n_short = np.sum(lengths <= short_upper)
    n_medium = np.sum((lengths > short_upper) & (lengths < long_lower))
    n_long = np.sum(lengths >= long_lower)

    print(f'\n{"=" * 55}')
    print(f'  建议的三分类边界 (基于 33% / 67% 分位数):')
    print(f'    Short  : ≤ {short_upper} words  →  {n_short} 条 ({n_short / total * 100:.1f}%)')
    if medium_lower <= medium_upper:
        print(f'    Medium : {medium_lower} – {medium_upper} words  →  {n_medium} 条 ({n_medium / total * 100:.1f}%)')
    else:
        print(f'    Medium : (空区间，边界重叠)')
    print(f'    Long   : ≥ {long_lower} words  →  {n_long} 条 ({n_long / total * 100:.1f}%)')

    # ── 同时给出中位数划分作为备选 ──
    med = int(p50)
    n_s_med = np.sum(lengths <= med)
    n_l_med = np.sum(lengths > med)
    print(f'\n  备选方案 — 中位数二分:')
    print(f'    ≤ {med} words: {n_s_med} 条 ({n_s_med / total * 100:.1f}%)')
    print(f'    > {med} words: {n_l_med} 条 ({n_l_med / total * 100:.1f}%)')

    # ── 绘制直方图 ──
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax1.set_xlabel('Word Count (per caption)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('HumanML3D Test Set — Text Prompt Length Distribution')
    ax1.axvline(np.median(lengths), color='red', linestyle='--', linewidth=1.5,
                label=f'median = {np.median(lengths):.0f}')
    ax1.axvline(p33, color='orange', linestyle=':', linewidth=1.5,
                label=f'33% = {p33:.0f}')
    ax1.axvline(p67, color='green', linestyle=':', linewidth=1.5,
                label=f'67% = {p67:.0f}')
    ax1.legend()

    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, 'eval_mdm_lego_histogram.pdf')
    fig.savefig(hist_path, dpi=150, format='pdf')
    print(f'\n  直方图已保存至: {hist_path}')

    # ── 保存统计信息 ──
    stats_path = os.path.join(OUTPUT_DIR, 'text_length_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'Total samples: {total}\n')
        f.write(f'Word count: min={lengths.min()}, max={lengths.max()}, '
                f'mean={lengths.mean():.2f}, median={np.median(lengths):.1f}, std={lengths.std():.2f}\n\n')
        for p in percentiles:
            f.write(f'{p}%: {np.percentile(lengths, p):.1f}\n')
        f.write(f'\nProposed bounds:\n')
        f.write(f'  Short  ≤ {short_upper}  ({n_short}, {n_short/total*100:.1f}%)\n')
        if medium_lower <= medium_upper:
            f.write(f'  Medium {medium_lower}-{medium_upper}  ({n_medium}, {n_medium/total*100:.1f}%)\n')
        f.write(f'  Long   ≥ {long_lower}  ({n_long}, {n_long/total*100:.1f}%)\n')
    print(f'  统计信息已保存至: {stats_path}')


if __name__ == '__main__':
    main()
