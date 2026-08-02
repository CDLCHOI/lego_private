"""
eval_mdm_lego_by_actionverbs.py
───────────────────────────────
实验目的：验证 LeGO 相对于 MDM，在 low / medium / high 三种不同动作复杂度下，
         FID 和 R_precision 是否都有提升。

边界（由 eval_mdm_lego_test_numverbs_histogram.py 的统计结果确定）：
    Low    : 1 verb      (1548 条, 35.3%) — 简单动作，如 "a person walks"
    Medium : 2 verbs     (1372 条, 31.3%) — 中等复杂度，如 "a man walks and waves"
    High   : ≥ 3 verbs   (1464 条, 33.4%) — 复杂动作，如 "a person walks forward while waving"

注: 有 44 条 (1.0%) caption 动词数为 0，归入 Low 组。

用法:
    python eval_mdm_lego_by_actionverbs.py              # 使用缓存
    python eval_mdm_lego_by_actionverbs.py --no_cache   # 忽略缓存，强制重新生成
"""

import os
import sys
import codecs as cs
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as pjoin

import options.option_transformer as option_trans
from utils.model_util import create_gaussian_diffusion_simple, get_mdm_bert_args
from utils.lora_util import load_lora_mdm_for_eval
from utils.mask_utils import generate_src_mask, load_ckpt
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.utils.metrics import (
    calculate_top_k,
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from dataset import dataset_control

# ═══════════════════════════════════════════════════════════════
# 超参数 / 路径常量
# ═══════════════════════════════════════════════════════════════

TEXT_DIR = './dataset/HumanML3D/texts'
TEST_FILE = './dataset/HumanML3D/test.txt'
DATA_ROOT = './dataset/HumanML3D'

# net1 (MDM, 无 LoRA) 和 net2 (LeGO, 有 CLIP LoRA) 的 checkpoint 路径
CKPT_MDM   = 'output/0814_MDMCLIP_b128/net_best.pth'
CKPT_LEGO  = 'output/0911_MDMCLIP_preatrainlora_ric1_b64/net_best.pth'
CKPT_LEGO2 = 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/net_best.pth'

BATCH_SIZE = 32
DIFFUSION_STEPS = 50
NUM_SAMPLES_LIMIT = 10000
OUTPUT_DIR = 'output/eval_by_actionverbs'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

# 由 eval_mdm_lego_test_numverbs_histogram.py 确定的分位数边界
# 动词分布：1 动词=35.3%, 2 动词=31.3%, ≥3 动词=33.4%
LOW_UPPER = 1     # Low  : ≤ 1 verb    (1592 条, 36.3%, 含 44 条 0-verb)
HIGH_LOWER = 3    # High : ≥ 3 verbs   (1464 条, 33.4%)
# Medium: 2 verbs                       (1372 条, 31.3%)


# ═══════════════════════════════════════════════════════════════
# 动作动词计数
# ═══════════════════════════════════════════════════════════════

def compute_action_verb_counts():
    """
    用 spaCy 对测试集每条 caption 进行 POS tagging，统计 VERB 数量。
    返回 {motion_id: verb_count}。
    """
    import spacy
    nlp = spacy.load('en_core_web_sm')

    with open(TEST_FILE, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    id_to_count = {}
    id_to_caption = {}
    all_captions = []

    for mid in test_ids:
        txt_path = pjoin(TEXT_DIR, f'{mid}.txt')
        try:
            with cs.open(txt_path, 'r') as f:
                first_line = f.readline().strip()
                caption = first_line.split('#')[0]
                id_to_caption[mid] = caption
                all_captions.append(caption)
        except Exception as e:
            print(f'  [警告] 读取 {txt_path} 失败: {e}')
            id_to_caption[mid] = ''

    # 批量处理更高效，但 spaCy 只能单个处理
    counts = []
    for cap in all_captions:
        doc = nlp(cap)
        n_verbs = sum(1 for token in doc if token.pos_ == 'VERB')
        counts.append(n_verbs)

    for mid, cnt in zip(test_ids, counts):
        id_to_count[mid] = cnt

    valid = [c for c in counts if c >= 0]
    print(f'[动词统计] 共 {len(valid)} 个有效样本, '
          f'min={min(valid)}, max={max(valid)}, '
          f'mean={np.mean(valid):.1f}, median={np.median(valid):.1f}')
    return id_to_count


def categorize_ids(id_to_count):
    """
    根据硬编码的边界 (LOW_UPPER=1, HIGH_LOWER=3) 将 motion ID 分为三组。
    """
    low_ids, medium_ids, high_ids = [], [], []

    for mid, vc in id_to_count.items():
        if vc <= LOW_UPPER:
            low_ids.append(mid)
        elif vc >= HIGH_LOWER:
            high_ids.append(mid)
        else:
            medium_ids.append(mid)

    total = len(id_to_count)
    print(f'[分类结果] Low (≤{LOW_UPPER} verb):     {len(low_ids)} 条 ({len(low_ids)/total*100:.1f}%)')
    print(f'[分类结果] Medium ({LOW_UPPER+1}-{HIGH_LOWER-1} verbs): {len(medium_ids)} 条 ({len(medium_ids)/total*100:.1f}%)')
    print(f'[分类结果] High (≥{HIGH_LOWER} verbs):   {len(high_ids)} 条 ({len(high_ids)/total*100:.1f}%)')
    return low_ids, medium_ids, high_ids


# ═══════════════════════════════════════════════════════════════
# 构建模型
# ═══════════════════════════════════════════════════════════════

def build_model(args, ckpt_path, use_lora):
    """
    参照 test_direction_speed.py 的 build_model：
    - use_lora=False: 用 load_ckpt 加载纯 MDM checkpoint
    - use_lora=True:  用 load_lora_mdm_for_eval 加载含 CLIP LoRA 的 checkpoint
    """
    from models.mdm_bert.mdm_bert import MDMBERT

    args.add_clip_lora = use_lora
    net = MDMBERT(**get_mdm_bert_args(args, 'mdm_bert'))

    if use_lora:
        load_lora_mdm_for_eval(net, ckpt_path)
    else:
        load_ckpt(net, ckpt_path, key=None, strict=False)

    diffusion = create_gaussian_diffusion_simple(args, net, 'mdm_bert')
    net.cuda()
    net.eval()
    return net, diffusion


# ═══════════════════════════════════════════════════════════════
# 创建类别筛选后的 DataLoader
# ═══════════════════════════════════════════════════════════════

def create_filtered_dataloader(args, category_name, id_list, mode='eval'):
    """
    基于给定 id_list 创建 DataLoader。
    将临时 split 文件写入 /tmp/（有写权限），使用绝对路径传入 split 参数。
    关键: os.path.join('a', '/abs/path') = '/abs/path'，所以传入绝对路径会覆盖 data_root。

    mode:
        'eval' — 训练空间 (Mean.npy/Std.npy)，用于生成 motion
        'gt'   — 评估器空间 (t2m_mean.npy/t2m_std.npy)，用于计算 GT embeddings
    返回 (loader, temp_split_path)
    """
    temp_split_path = os.path.join('/tmp', f't2m_test_byverbs_{category_name}.txt')
    with open(temp_split_path, 'w') as f:
        for mid in id_list:
            f.write(mid + '\n')

    # 去除 .txt 后缀作为 split 参数（传入绝对路径，pjoin 会自动忽略 data_root）
    split_name = temp_split_path.replace('.txt', '')

    print(f'[临时 split] {temp_split_path}  共 {len(id_list)} 条  mode={mode}')

    loader = dataset_control.DataLoader(
        batch_size=args.batch_size,
        args=args,
        mode=mode,
        split=split_name,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    return loader, temp_split_path


# ═══════════════════════════════════════════════════════════════
# 生成 motion（带磁盘缓存）
# ═══════════════════════════════════════════════════════════════

def generate_motions_from_loader(args, gen_loader, diffusion,
                                  cache_path=None, num_samples_limit=NUM_SAMPLES_LIMIT):
    """
    遍历 gen_loader，用 diffusion.p_sample_loop 生成 motion。
    如果 cache_path 存在，直接从缓存加载，跳过扩散生成。

    返回 list[dict]，每个 dict 与 CompADCGeneratedDataset.generated_motion 格式兼容：
        {'motion': (196,263) float32, 'length': scalar, 'caption': str,
         'tokens': str, 'cap_len': int, 'filename': str}
    """
    # ── 检查磁盘缓存 ──
    if cache_path and os.path.exists(cache_path):
        print(f'  [缓存命中] 加载 {cache_path}')
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    generated_motion = []
    real_num_batches = len(gen_loader)

    with torch.no_grad():
        for i, batch in enumerate(gen_loader):
            if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                break
            print(f'  [生成] batch {i+1}/{real_num_batches}')

            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, \
                txt_tokens, traj, traj_mask_263, traj_mask, filename = batch

            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            real_mask = generate_src_mask(max_length, real_length)

            model_kwargs = {
                'gt_motion': gt_motion,
                'real_mask': real_mask,
                'clip_text': clip_text,
                'word_embs': word_embeddings.float().cuda(),
                'pos_ohot': pos_one_hots.float().cuda(),
                'cap_lens': sent_len.cuda(),
                'real_length': real_length,
            }

            sample = diffusion.p_sample_loop(
                None, with_control=True, model_kwargs=model_kwargs, batch_size=b
            )

            # ── 反归一化（训练空间 → 原始空间）→ 重新归一化到评估器空间 ──
            gen_dataset = gen_loader.dataset
            t2m_dataset = gen_dataset.t2m_dataset

            for bs_i in range(b):
                motion = sample[bs_i].cpu().numpy()                      # (196, 263) 训练归一化空间
                denormed = t2m_dataset.inv_transform(motion)             # 反归一化 → 原始空间
                dim = gen_dataset.mean_for_eval.shape[0]                 # 通常是 263
                renormed = (denormed[:, :dim] - gen_dataset.mean_for_eval) / gen_dataset.std_for_eval

                generated_motion.append({
                    'motion': renormed.astype(np.float32),
                    'length': real_length[bs_i].cpu().numpy(),
                    'caption': clip_text[bs_i],
                    'tokens': txt_tokens[bs_i],
                    'cap_len': sent_len[bs_i].item(),
                    'filename': filename[bs_i],
                })

    print(f'  [生成] 完成，共 {len(generated_motion)} 条')

    # ── 写入磁盘缓存 ──
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(generated_motion, cache_path)
        print(f'  [缓存] 已保存至 {cache_path}')

    return generated_motion


# ═══════════════════════════════════════════════════════════════
# GeneratedDataset（将生成数据包装为 evaluator 兼容格式）
# ═══════════════════════════════════════════════════════════════

class GeneratedDataset(torch.utils.data.Dataset):
    """
    将 generate_motions_from_loader 产出的数据包装成 Dataset，
    __getitem__ 格式与 CompADCGeneratedDataset 一致。
    """

    def __init__(self, generated_motion, w_vectorizer):
        self.generated_motion = generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion = data['motion']
        m_length = data['length']
        caption = data['caption']
        tokens = data['tokens']
        sent_len = data['cap_len']
        hint = np.zeros((motion.shape[0], 22, 3), dtype=np.float32)
        filename = data.get('filename', '')

        # token 字符串 → word_embeddings / pos_one_hots
        if isinstance(tokens, str):
            tokens = tokens.split('_')

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length,
                '_'.join(tokens), hint, filename)


def collate_sort_by_length(batch):
    """按 sent_len 降序排列（与原 collate_fn 一致）"""
    batch.sort(key=lambda x: x[3], reverse=True)
    return torch.utils.data._utils.collate.default_collate(batch)


# ═══════════════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════════════

def evaluate_r_precision_and_collect_embeddings(eval_wrapper, motion_loader):
    """
    单次遍历 motion_loader，同时完成：
    1. 计算 R_precision（余弦相似度 top-1/2/3）
    2. 收集所有 motion embeddings（用于后续 FID 计算）
    返回 (r_precision_ndarray, all_motion_embeddings)
    """
    all_size = 0
    topk_count_sim_sum = 0
    all_motion_embeddings = []

    with torch.no_grad():
        for batch in motion_loader:
            if len(batch) == 7:
                word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _ = batch
            else:
                word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, filename = batch

            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=m_lens,
            )

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            # 余弦相似度 → top-k 检索精度
            text_embeds = F.normalize(text_embeddings, dim=-1)
            motion_embeds = F.normalize(motion_embeddings, dim=-1)
            sim_mat = text_embeds @ motion_embeds.T
            argsmax_sim = np.argsort((0 - sim_mat).cpu().numpy(), axis=1)
            top_k_sim_mat = calculate_top_k(argsmax_sim, top_k=3)
            top_k_sim_count = top_k_sim_mat.sum(axis=0)
            topk_count_sim_sum += top_k_sim_count
            all_size += text_embeddings.shape[0]

    r_precision = topk_count_sim_sum / all_size
    all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
    return r_precision, all_motion_embeddings


def compute_gt_embeddings(eval_wrapper, gt_loader):
    """
    从 gt_loader 中提取所有 GT motion 的 embeddings，
    返回 (gt_mu, gt_cov) 用于 FID 计算。
    """
    gt_embeddings = []
    with torch.no_grad():
        for batch in gt_loader:
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _, filename = batch
            emb = eval_wrapper.get_motion_embeddings(motions=motions, m_lens=m_lens)
            gt_embeddings.append(emb.cpu().numpy())
    gt_embeddings = np.concatenate(gt_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    return gt_mu, gt_cov


def compute_fid_from_embeddings(gen_embeddings, gt_mu, gt_cov):
    """从已收集的 generated embeddings 和 GT 统计量计算 FID"""
    mu, cov = calculate_activation_statistics(gen_embeddings)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    return fid


# ═══════════════════════════════════════════════════════════════
# 单类别完整评估
# ═══════════════════════════════════════════════════════════════

def evaluate_category(args, category_name, category_ids, net1, diffusion1,
                      net2, diffusion2, net3, diffusion3, eval_wrapper, no_cache=False):
    """
    对单个动作复杂度类别进行评估：
    1. 创建 gt_loader 和 gen_loader
    2. 用 MDM (net1) 生成 motion → 计算 R_precision / FID
    3. 用 LeGO (net2) 生成 motion → 计算 R_precision / FID
    4. 用 LeGO2 (net3) 生成 motion → 计算 R_precision / FID
    """
    print(f'\n{"=" * 60}')
    print(f'[评估] 类别: {category_name}  样本数: {len(category_ids)}')
    print(f'{"=" * 60}')

    if len(category_ids) < BATCH_SIZE:
        print(f'[警告] {category_name} 样本数 ({len(category_ids)}) < batch_size ({BATCH_SIZE})，跳过！')
        return {'R_precision': None, 'FID': None}, {'R_precision': None, 'FID': None}

    # ── 创建 DataLoader ──
    # GT: mode='gt' → 评估器空间 (t2m_mean.npy/t2m_std.npy)
    # Gen: mode='eval' → 训练空间 (Mean.npy/Std.npy)，生成后手动 renorm 到评估器空间
    gt_loader, _ = create_filtered_dataloader(args, category_name, category_ids, mode='gt')
    gen_loader, _ = create_filtered_dataloader(args, category_name, category_ids, mode='eval')

    w_vectorizer = gen_loader.dataset.w_vectorizer

    # ── 预计算 GT embeddings（所有模型共享） ──
    print(f'\n  >>> [{category_name}] 预计算 GT embeddings ...')
    gt_mu, gt_cov = compute_gt_embeddings(eval_wrapper, gt_loader)
    print(f'  >>> [{category_name}] GT embeddings 计算完成')

    results = {}

    # ── 分别对 MDM、LeGO、LeGO2 生成并评估 ──
    for model_name, net, diffusion in [
        ('MDM', net1, diffusion1),
        ('LeGO', net2, diffusion2),
        ('LeGO2', net3, diffusion3),
    ]:
        model_tag = f'{category_name}_{model_name}'
        cache_path = os.path.join(CACHE_DIR, f'{model_name}_{category_name}_motions.pt') if not no_cache else None
        if no_cache:
            cache_path = None

        print(f'\n  >>> [{model_tag}] 生成/加载 motion ...')
        gen_motions = generate_motions_from_loader(
            args, gen_loader, diffusion,
            cache_path=cache_path,
            num_samples_limit=min(len(category_ids), NUM_SAMPLES_LIMIT),
        )

        # 构造 GeneratedDataset → DataLoader
        gen_dataset = GeneratedDataset(gen_motions, w_vectorizer)
        gen_motion_loader = torch.utils.data.DataLoader(
            gen_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_sort_by_length,
            drop_last=True,
            num_workers=0,
        )

        # 单次遍历：同时计算 R_precision 和收集 motion embeddings
        print(f'  >>> [{model_tag}] 计算 R_precision & 收集 embeddings ...')
        r_precision, gen_embeddings = evaluate_r_precision_and_collect_embeddings(
            eval_wrapper, gen_motion_loader
        )
        print(f'  >>> [{model_tag}] R_precision: top1={r_precision[0]:.4f}, '
              f'top2={r_precision[1]:.4f}, top3={r_precision[2]:.4f}')

        # 计算 FID
        print(f'  >>> [{model_tag}] 计算 FID ...')
        fid = compute_fid_from_embeddings(gen_embeddings, gt_mu, gt_cov)
        print(f'  >>> [{model_tag}] FID: {fid:.4f}')

        results[model_name] = {
            'R_precision': r_precision,
            'FID': fid,
        }

        # 释放内存
        del gen_motions, gen_dataset, gen_motion_loader, gen_embeddings
        torch.cuda.empty_cache()

    return results['MDM'], results['LeGO'], results['LeGO2']


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 解析命令行参数 ──
    no_cache = '--no_cache' in sys.argv
    if no_cache:
        sys.argv.remove('--no_cache')

    args = option_trans.get_args_parser()

    # ── 固定关键参数 ──
    args.dataset_name = 't2m'
    args.modeltype = 'mdm_bert'
    args.text_encoder_type = 'clip'
    args.batch_size = BATCH_SIZE
    args.diffusion_steps = DIFFUSION_STEPS
    # 注意: no_random=True 会触发 ControlDataset.__init__ 中的 id_list[:111] 截断!
    # 必须设为 False 才能加载全部样本。用 seed 保证可复现。
    args.no_random = False
    args.cond_mode = 'both_text_spatial'
    args.guidance_param = 2.5
    args.normalize_traj = True
    args.density = 0
    args.control_joint = [-1]
    args.multi_joint_control = False
    args.unit_length = 4
    args.down_t = 2
    args.max_motion_length = 196
    args.evaluator_eval = None
    args.evaluator_train = None
    args.using_meta = False
    args.train_sample_num = 0
    args.seed = 0
    args.max_samples = NUM_SAMPLES_LIMIT
    args.eval_mode = 'no_mm'
    args.replication_times = 1
    args.timestep_respacing = '100'
    args.use_ddim = 0
    args.return_type = 'sample'

    # 设置全局随机种子，保证可复现
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not no_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════
    # 1. 动作动词分类（硬编码边界）
    # ═══════════════════════════════════════════════════
    print('\n' + '#' * 60)
    print('#  1. 动作动词分类')
    print('#' * 60)
    print(f'  边界: Low ≤ {LOW_UPPER} verb, Medium {LOW_UPPER+1}-{HIGH_LOWER-1} verbs, High ≥ {HIGH_LOWER} verbs')

    id_to_count = compute_action_verb_counts()
    low_ids, medium_ids, high_ids = categorize_ids(id_to_count)

    # ═══════════════════════════════════════════════════
    # 2. 载入模型 & 评估器
    # ═══════════════════════════════════════════════════
    print('\n' + '#' * 60)
    print('#  2. 载入模型 & 评估器')
    print('#' * 60)

    print('\n[2.1] 构建 MDM (net1, 无 CLIP LoRA) ...')
    net1, diffusion1 = build_model(args, CKPT_MDM, use_lora=False)

    print('[2.2] 构建 LeGO (net2, 有 CLIP LoRA) ...')
    net2, diffusion2 = build_model(args, CKPT_LEGO, use_lora=True)

    print('[2.3] 构建 LeGO2 (net3, 有 CLIP LoRA, scratch) ...')
    net3, diffusion3 = build_model(args, CKPT_LEGO2, use_lora=True)

    print('[2.4] 构建 Evaluator (GRU) ...')
    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args, args.evaluator_eval)

    # ═══════════════════════════════════════════════════
    # 3. 逐类别评估
    # ═══════════════════════════════════════════════════
    print('\n' + '#' * 60)
    print('#  3. 逐类别评估 (no_cache={})'.format(no_cache))
    print('#' * 60)

    all_results = {}

    for cat_name, cat_ids in [
        ('low', low_ids),
        ('medium', medium_ids),
        ('high', high_ids),
    ]:
        mdm_res, lego_res, lego2_res = evaluate_category(
            args, cat_name, cat_ids,
            net1, diffusion1, net2, diffusion2, net3, diffusion3, eval_wrapper,
            no_cache=no_cache,
        )
        all_results[cat_name] = {'MDM': mdm_res, 'LeGO': lego_res, 'LeGO2': lego2_res}

    # ═══════════════════════════════════════════════════
    # 4. 汇总输出
    # ═══════════════════════════════════════════════════
    print('\n' + '#' * 60)
    print('#  最终结果汇总')
    print('#' * 60)

    header = f"{'Category':<10} {'Model':<8} {'R@1':<8} {'R@2':<8} {'R@3':<8} {'FID':<8}"
    sep = '-' * 50
    print(header)
    print(sep)

    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(header + '\n')
        f.write(sep + '\n')

        for cat_name in ['low', 'medium', 'high']:
            for model_name in ['MDM', 'LeGO', 'LeGO2']:
                res = all_results[cat_name][model_name]
                if res['R_precision'] is not None:
                    rp = res['R_precision']
                    line = (f"{cat_name:<10} {model_name:<8} "
                            f"{rp[0]:.4f}  {rp[1]:.4f}  {rp[2]:.4f}  {res['FID']:.4f}")
                else:
                    line = f"{cat_name:<10} {model_name:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}"
                print(line)
                f.write(line + '\n')

    print(f'\n汇总结果已保存至: {summary_path}')
    print('Done!')


if __name__ == '__main__':
    main()
