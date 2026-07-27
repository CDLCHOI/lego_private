import collections
import torch
import numpy as np
from torch.utils import data
from os.path import join as pjoin
import random
from tqdm import tqdm
import json


class CommonMotionDataset(data.Dataset):
    """SnapMoGen 通用动作数据集基类"""
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, is_debug=False):
        self.cfg = cfg
        mid_list = []
        cid_list = []
        total_frames = 0

        data_dict = {}

        with open(mid_list_path, "r") as f:
            for line in f.readlines():
                mid_list.append(line.strip())

        with open(cid_list_path, "r") as f:
            for line in f.readlines():
                cid = line.strip()
                _, start, end = cid.split("#")

                if int(end) - int(start) >= cfg.data.min_motion_length:
                    cid_list.append(cid)
                    total_frames += int(end) - int(start)

        total_count = len(cid_list)

        if is_debug:
            cid_list = cid_list[:333]
            needed_mids = set(cid.split('#')[0] for cid in cid_list)
            mid_list = [m for m in mid_list if m in needed_mids]
            total_count = len(cid_list)
            print(f'[DEBUG] Truncated to {total_count} cids, {len(mid_list)} motions')
        for i, mid in tqdm(enumerate(mid_list)):
            data_path = pjoin(cfg.data.feat_dir, "%s.npy" % mid)
            data = np.load(data_path)
            data_dict[mid] = data

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.cfg = cfg
        self.mid_list = mid_list
        self.cid_list = cid_list

        print(
            "Loading %d motions, %d frames, %03f hours"
            % (total_count, total_frames, total_frames / 30.0 / 60.0 / 60.0)
        )

    def inv_transform(self, data):
        if isinstance(data, np.ndarray):
            return data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        elif isinstance(data, torch.Tensor):
            return data * torch.from_numpy(self.std[:data.shape[-1]]).float().to(
                data.device
            ) + torch.from_numpy(self.mean[:data.shape[-1]]).float().to(data.device)
        else:
            raise TypeError("Expected data to be either np.ndarray or torch.Tensor")

    def __len__(self):
        return len(self.cid_list)

    def __getitem__(self, item):
        cid = self.cid_list[item]
        mid, start, end = cid.split("#")
        motion = self.data_dict[mid][int(start) : int(end)]

        # Z Normalization
        motion_data = (motion - self.mean) / self.std

        return motion_data, cid


class TextMotionDataset(CommonMotionDataset):
    """SnapMoGen 文本-动作数据集

    支持两种模式：
    1. 基础模式（无 w_vectorizer）：返回 (caption, motion, m_length)
    2. 文本向量化模式（提供 w_vectorizer）：返回 (word_embeddings, pos_one_hots, caption,
       sent_len, motion, m_length, tokens_str)
    """
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path,
                 w_vectorizer=None, opt=None, is_debug=False):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path, is_debug=is_debug)

        with open(all_caption_path, "r") as f:
            self.all_captions = json.load(f)

        self.w_vectorizer = w_vectorizer
        self.opt = opt

        if w_vectorizer is not None:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')

    def __getitem__(self, item):
        # ── 1. 加载运动数据（已完成 Z-normalization） ──
        motion, cid = super().__getitem__(item)

        # ── 2. 加载文本标注 ──
        captions = self.all_captions[cid]["manual"] + self.all_captions[cid]["gpt"]
        caption = random.choice(captions)

        # ── 3. 运动长度处理 ──
        # 确定 unit_length：优先使用 opt.unit_length，否则使用 cfg.data.unit_length
        unit_length = self.opt.unit_length if self.opt is not None else self.cfg.data.unit_length

        m_length = (
            len(motion)
            if len(motion) < self.cfg.data.max_motion_length
            else self.cfg.data.max_motion_length
        )

        m_length = (m_length // unit_length) * unit_length

        # 随机裁剪
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]

        # Zero-padding 到 max_motion_length
        if m_length < self.cfg.data.max_motion_length:
            motion = np.concatenate(
                [motion,
                 np.zeros((self.cfg.data.max_motion_length - m_length, motion.shape[1]))],
                axis=0,
            )

        # ── 4. 文本向量化（如果提供了 w_vectorizer） ──
        if self.w_vectorizer is not None:
            # spacy tokenization
            doc = self.nlp(caption)

            tokens = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                    tokens.append(token.lemma_ + '/' + token.pos_)
                else:
                    tokens.append(word + '/' + token.pos_)

            # 截断/填充到 opt.max_text_len + 2（sos + eos）
            if len(tokens) < self.opt.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)

            # 通过 w_vectorizer 转换为 word_embeddings 和 pos_one_hots
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

        # 基础模式：不进行文本向量化
        return caption, motion, m_length


class MotionDataset(CommonMotionDataset):
    """SnapMoGen 纯动作数据集（无文本）"""
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)
        lengths = [0]
        n_cid_list = []
        for cid in self.cid_list:
            _, start, end = cid.split("#")
            length = int(end) - int(start) - self.cfg.data.motion_length
            if length >= 0:
                lengths.append(length)
                n_cid_list.append(cid)

        self.cid_list = n_cid_list
        self.cumsum = np.cumsum(lengths)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        cid_idx = np.searchsorted(self.cumsum, item + 1) - 1
        idx = item - self.cumsum[cid_idx]
        motion, _ = super().__getitem__(cid_idx)
        motion_clip = motion[idx : idx + self.cfg.data.motion_length]

        return motion_clip
