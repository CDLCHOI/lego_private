"""
eval_mdm_lego_test_numverbs_histogram.py
───────────────────────────
独立脚本：统计 HumanML3D 测试集所有文本 prompt 中动作动词（action verbs）的数量分布，
绘制直方图，并打印每个动词数量的样本数量和占比。
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
OUTPUT_DIR = 'output/eval_by_actionverbs'


# ═══════════════════════════════════════════════════════════════
# 动词计数
# ═══════════════════════════════════════════════════════════════

def count_action_verbs_with_spacy(captions):
    """使用 spaCy 对每条 caption 进行 POS tagging，统计 VERB 数量。"""
    import spacy
    nlp = spacy.load('en_core_web_sm')
    verb_counts = []
    for cap in captions:
        doc = nlp(cap)
        n_verbs = sum(1 for token in doc if token.pos_ == 'VERB')
        verb_counts.append(n_verbs)
    return np.array(verb_counts)


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 读取测试集文件列表 ──
    with open(TEST_FILE, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    print(f'测试集 motion 文件总数: {len(test_ids)}')

    # ── 逐个读取文本第一行，收集所有 captions ──
    captions = []
    skipped = 0
    for mid in test_ids:
        txt_path = pjoin(TEXT_DIR, f'{mid}.txt')
        try:
            with cs.open(txt_path, 'r') as f:
                first_line = f.readline().strip()
                caption = first_line.split('#')[0]
                captions.append(caption)
        except Exception as e:
            print(f'  [警告] 读取 {txt_path} 失败: {e}')
            skipped += 1

    print(f'  有效文本数: {len(captions)}, 跳过: {skipped}')

    # ── 用 spaCy 统计每条 caption 的动作动词数 ──
    print(f'  正在用 spaCy 进行 POS tagging ({len(captions)} 条 caption) ...')
    verb_counts = count_action_verbs_with_spacy(captions)

    # ── 基本统计 ──
    print(f'\n{"=" * 55}')
    print(f'  有效文本数: {len(verb_counts)}')
    print(f'  动词数范围: {verb_counts.min()} – {verb_counts.max()}')
    print(f'  均值: {verb_counts.mean():.2f}')
    print(f'  中位数: {np.median(verb_counts):.1f}')
    print(f'  标准差: {verb_counts.std():.2f}')

    # ── 分位数 ──
    percentiles = [5, 10, 20, 25, 33, 50, 67, 75, 80, 90, 95]
    print(f'\n{"=" * 55}')
    print(f'  分位数统计:')
    for p in percentiles:
        val = np.percentile(verb_counts, p)
        print(f'    {p:3d}%: {val:6.1f} verbs')

    # ── 每个动词数量的详细统计 ──
    counter = Counter(verb_counts)
    print(f'\n{"=" * 55}')
    print(f'  每个动词数的样本数量及占比（按动词数升序）:')
    print(f'  {"动词数":>6s}  {"样本数":>6s}  {"占比":>8s}  {"累积占比":>8s}')
    print(f'  {"-" * 35}')

    cumulative = 0
    total = len(verb_counts)
    for vc in sorted(counter.keys()):
        count = counter[vc]
        cumulative += count
        pct = count / total * 100
        cum_pct = cumulative / total * 100
        print(f'  {vc:6d}  {count:6d}  {pct:7.2f}%  {cum_pct:7.2f}%')

    # ── 33% / 67% 分位数边界 ──
    p33 = np.percentile(verb_counts, 33)
    p50 = np.median(verb_counts)
    p67 = np.percentile(verb_counts, 67)

    low_upper = int(np.floor(p33))
    medium_lower = low_upper + 1
    medium_upper_val = int(np.ceil(p67)) - 1
    high_lower = int(np.ceil(p67))

    # 防止边界重叠
    if low_upper >= high_lower:
        low_upper = int(p50)
        high_lower = low_upper + 1
        medium_lower = low_upper + 1
        medium_upper_val = low_upper
        print(f'\n  ⚠ 33%/67% 分位数边界重叠，回退到中位数 {int(p50)} 划分')

    n_low = np.sum(verb_counts <= low_upper)
    n_medium = np.sum((verb_counts > low_upper) & (verb_counts < high_lower))
    n_high = np.sum(verb_counts >= high_lower)

    print(f'\n{"=" * 55}')
    print(f'  建议的三分类边界 (基于 33% / 67% 分位数):')
    print(f'    Low    : ≤ {low_upper} verb(s)  →  {n_low} 条 ({n_low / total * 100:.1f}%)')
    if medium_lower <= medium_upper_val:
        print(f'    Medium : {medium_lower} – {medium_upper_val} verbs  →  {n_medium} 条 ({n_medium / total * 100:.1f}%)')
    else:
        print(f'    Medium : (空区间，边界重叠)')
    print(f'    High   : ≥ {high_lower} verbs  →  {n_high} 条 ({n_high / total * 100:.1f}%)')

    # ── 同时给出中位数划分作为备选 ──
    med = int(p50)
    n_low_med = np.sum(verb_counts <= med)
    n_high_med = np.sum(verb_counts > med)
    print(f'\n  备选方案 — 中位数二分:')
    print(f'    ≤ {med} verb(s): {n_low_med} 条 ({n_low_med / total * 100:.1f}%)')
    print(f'    > {med} verb(s): {n_high_med} 条 ({n_high_med / total * 100:.1f}%)')

    # ── 绘制直方图 ──
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.arange(verb_counts.min(), verb_counts.max() + 2) - 0.5
    ax1.hist(verb_counts, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    ax1.set_xlabel('Number of Action Verbs (per caption)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('HumanML3D Test Set — Action Verb Count Distribution')
    ax1.axvline(np.median(verb_counts), color='red', linestyle='--', linewidth=1.5,
                label=f'median = {np.median(verb_counts):.0f}')
    ax1.axvline(p33, color='orange', linestyle=':', linewidth=1.5,
                label=f'33% = {p33:.0f}')
    ax1.axvline(p67, color='green', linestyle=':', linewidth=1.5,
                label=f'67% = {p67:.0f}')
    ax1.legend()
    ax1.set_xticks(np.arange(verb_counts.min(), verb_counts.max() + 1))

    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, 'action_verbs_histogram.pdf')
    fig.savefig(hist_path, dpi=150, format='pdf')
    print(f'\n  直方图已保存至: {hist_path}')

    # ── 保存统计信息 ──
    stats_path = os.path.join(OUTPUT_DIR, 'action_verbs_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'Total samples: {total}\n')
        f.write(f'Verb count: min={verb_counts.min()}, max={verb_counts.max()}, '
                f'mean={verb_counts.mean():.2f}, median={np.median(verb_counts):.1f}, std={verb_counts.std():.2f}\n\n')
        for p in percentiles:
            f.write(f'{p}%: {np.percentile(verb_counts, p):.1f}\n')
        f.write(f'\nProposed bounds:\n')
        f.write(f'  Low    ≤ {low_upper}  ({n_low}, {n_low/total*100:.1f}%)\n')
        if medium_lower <= medium_upper_val:
            f.write(f'  Medium {medium_lower}-{medium_upper_val}  ({n_medium}, {n_medium/total*100:.1f}%)\n')
        f.write(f'  High   ≥ {high_lower}  ({n_high}, {n_high/total*100:.1f}%)\n')
    print(f'  统计信息已保存至: {stats_path}')


if __name__ == '__main__':
    main()
