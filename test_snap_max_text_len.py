"""
测试 SnapMoGen 数据集文本最大单词数量。

从 all_caption_clean.json 读取所有 caption，统计：
1. 原始 split 单词数分布
2. spacy tokenization + alpha-filter 后的单词数分布（模拟 dataset 实际行为）
"""

import json
import sys

DATA_PATH = '/data/motion/SnapMoGen/all_caption_clean.json'

with open(DATA_PATH, 'r') as f:
    data = json.load(f)

# ── 1. 原始 split 统计 ──
raw_counts = []
max_raw = 0
max_raw_cap = ''
for cid, val in data.items():
    for cap in val['manual'] + val['gpt']:
        wc = len(cap.split())
        raw_counts.append(wc)
        if wc > max_raw:
            max_raw = wc
            max_raw_cap = cap

raw_counts.sort()
total = len(raw_counts)

print("=" * 60)
print("原始 split 单词数统计:")
print(f"  总 caption 数: {total}")
print(f"  min: {raw_counts[0]}")
print(f"  50%: {raw_counts[int(total * 0.50)]}")
print(f"  90%: {raw_counts[int(total * 0.90)]}")
print(f"  95%: {raw_counts[int(total * 0.95)]}")
print(f"  99%: {raw_counts[int(total * 0.99)]}")
print(f"  max: {raw_counts[-1]}")
print(f"  >20 单词: {sum(1 for w in raw_counts if w > 20)}/{total} ({100*sum(1 for w in raw_counts if w > 20)/total:.1f}%)")
print(f"  >22 单词: {sum(1 for w in raw_counts if w > 22)}/{total} ({100*sum(1 for w in raw_counts if w > 22)/total:.1f}%)")
print(f"  >77 单词: {sum(1 for w in raw_counts if w > 77)}/{total} ({100*sum(1 for w in raw_counts if w > 77)/total:.1f}%)")
print(f"  >120 单词: {sum(1 for w in raw_counts if w > 120)}/{total} ({100*sum(1 for w in raw_counts if w > 120)/total:.1f}%)")
print()
print(f"  最长 caption ({max_raw} 单词): {max_raw_cap[:200]}...")

# ── 2. spacy tokenization 统计（如果可用） ──
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("\n" + "=" * 60)
    print("spacy tokenization + alpha-filter 单词数统计:")

    spacy_counts = []
    max_spacy = 0
    max_spacy_cap = ''
    for cid, val in data.items():
        for cap in val['manual'] + val['gpt']:
            doc = nlp(cap)
            tokens = [token.text for token in doc if token.text.isalpha()]
            wc = len(tokens)
            spacy_counts.append(wc)
            if wc > max_spacy:
                max_spacy = wc
                max_spacy_cap = cap

    spacy_counts.sort()
    total_s = len(spacy_counts)

    print(f"  总 caption 数: {total_s}")
    print(f"  min: {spacy_counts[0]}")
    print(f"  50%: {spacy_counts[int(total_s * 0.50)]}")
    print(f"  90%: {spacy_counts[int(total_s * 0.90)]}")
    print(f"  95%: {spacy_counts[int(total_s * 0.95)]}")
    print(f"  99%: {spacy_counts[int(total_s * 0.99)]}")
    print(f"  max: {spacy_counts[-1]}")
    print(f"  >20 单词: {sum(1 for w in spacy_counts if w > 20)}/{total_s} ({100*sum(1 for w in spacy_counts if w > 20)/total_s:.1f}%)")
    print(f"  >22 单词: {sum(1 for w in spacy_counts if w > 22)}/{total_s} ({100*sum(1 for w in spacy_counts if w > 22)/total_s:.1f}%)")
    print()
    print(f"  最长 caption ({max_spacy} 单词): {max_spacy_cap[:200]}...")

except ImportError:
    print("\n[INFO] spacy 不可用，跳过 spacy tokenization 统计")
    print("[INFO] 可以用以下命令安装: pip install spacy && python -m spacy download en_core_web_sm")

# ── 3. 与关键阈值对比 ──
print("\n" + "=" * 60)
print("关键阈值对比:")
print(f"  evaluator 训练 max_text_len:  20 (+2 sos/eos = 22)")
print(f"  CLIP text encoder max tokens:  77")
print(f"  evaluator.yaml max_text_length: 120")
print(f"  数据集 max raw words:          {raw_counts[-1]}")
print(f"  被截断到20单词的比例:           {100*sum(1 for w in raw_counts if w > 20)/total:.1f}%")
