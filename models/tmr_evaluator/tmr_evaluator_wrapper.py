"""
TMR (Text-to-Motion Retrieval) Evaluator Wrapper

将预训练的 TMR 模型封装为与现有训练管线兼容的 evaluator 接口。

用法:
    wrapper = TMREvaluatorWrapper(ckpt_dir="TMR/models/models/tmr_humanml3d_guoh3dfeats", device="cuda")
    text_latent, _ = wrapper.encode_text(["a person walks forward"], sample_mean=True)
    motion_latent, _ = wrapper.encode_motion(motion, lengths, sample_mean=False)

归一化域转换:
    TMR 预训练模型使用 TMR 自己的 mean/std 做归一化（stats/humanml3d/guoh3dfeats/），
    而扩散模型输出的是 HumanML3D 归一化域（dataset/HumanML3D/Mean.npy, Std.npy）。
    本 wrapper 在 encode_motion 时会自动做归一化域转换：
    motion_tmr = (motion_hml * Std_hml + Mean_hml - Mean_tmr) / Std_tmr
"""
import os

import torch
import torch.nn as nn
import numpy as np

from models.tmr_evaluator.modules import ACTORStyleEncoder, length_to_mask


class DistilBERTTextEncoder(nn.Module):
    """
    DistilBERT 文本编码器（冻结权重，仅用于将 raw text → token embeddings）
    默认使用 distilbert-base-uncased
    """

    def __init__(self, modelpath: str = "distilbert-base-uncased", device: str = "cpu"):
        super().__init__()

        self.device = device
        from transformers import AutoTokenizer, AutoModel
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)
        self.text_encoded_dim = self.text_model.config.hidden_size  # 768

        # 冻结所有权重，始终处于 eval 模式
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.to(device)

    def train(self, mode: bool = True) -> nn.Module:
        """始终返回 eval 模式，DistilBERT 不参与训练"""
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts):
        """
        Args:
            texts: str 或 List[str]
        Returns:
            dict: {"x": (B, max_len, 768), "length": (B,)} — token-level embeddings + lengths
        """
        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.device))
        length = encoded_inputs.attention_mask.to(dtype=bool).sum(1)

        if squeeze:
            return {"x": output.last_hidden_state[0], "length": length[0]}
        return {"x": output.last_hidden_state, "length": length}


class TMREvaluatorWrapper(nn.Module):
    """
    TMR Evaluator 封装类

    与现有训练管线兼容，提供:
    - encode_text(text_list, sample_mean) → (latent_vec, dists)
    - encode_motion(motion, lengths, sample_mean) → (latent_vec, dists)
    - get_motion_embeddings_with_grad(motions, m_lens) → motion_emb
    - get_co_embeddings_with_grad(text_list, motions, m_lens) → (text_emb, motion_emb)
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: torch.device,
        motion_nfeats: int = 263,
        text_nfeats: int = 768,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        distilbert_modelname: str = "distilbert-base-uncased",
        hml_mean_std_dir: str = "dataset/HumanML3D",
        tmr_stats_dir: str = None,
    ):
        """
        Args:
            ckpt_dir: TMR 预训练权重目录 (包含 last_weights/ 子目录)
            device: 运算设备
            motion_nfeats: motion 特征维度 (HumanML3D guoh3dfeats = 263)
            text_nfeats: text token 特征维度 (DistilBERT = 768)
            latent_dim: TMR 潜在空间维度
            ff_size, num_layers, num_heads, dropout, activation: ACTORStyleEncoder 参数
            distilbert_modelname: DistilBERT 模型名称
            hml_mean_std_dir: HumanML3D 项目的 Mean.npy / Std.npy 所在目录
            tmr_stats_dir: TMR 的 mean.pt / std.pt 所在目录，默认自动从 ckpt_dir 推导
        """
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.dim_pose = motion_nfeats  # 兼容现有 eval 管线的 dim_pose 属性
        self._debug_norm_printed = False  # 调试用：只打印一次归一化信息

        # ---- 加载归一化统计量 ----
        self._load_normalization_stats(hml_mean_std_dir, ckpt_dir, tmr_stats_dir, device)

        # 归一化域转换: 本项目有两个不同的归一化域
        #   - 'hml': dataset/HumanML3D/Mean.npy + Std.npy (训练数据 mode='train')
        #   - 'meta': dataset/t2m_mean.npy + t2m_std.npy (GT评估数据 mode='gt')
        self._norm_domain = 'hml'  # 默认使用 HML 域 (训练用)
        self._input_mean = self._hml_mean
        self._input_std = self._hml_std

        # ---- 构建 motion encoder (ACTORStyleEncoder) ----
        self.motion_encoder = ACTORStyleEncoder(
            nfeats=motion_nfeats,
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )

        # ---- 构建 text encoder (ACTORStyleEncoder) ----
        self.text_encoder = ACTORStyleEncoder(
            nfeats=text_nfeats,
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )

        # ---- 构建 DistilBERT token embedder (冻结) ----
        self.distilbert = DistilBERTTextEncoder(
            modelpath=distilbert_modelname, device=str(device)
        )

        # ---- 加载预训练权重 ----
        self._load_pretrained_weights(ckpt_dir)

        self.to(device)

    def _load_normalization_stats(self, hml_mean_std_dir, ckpt_dir, tmr_stats_dir, device):
        """加载三套归一化统计量，用于 motion 归一化域转换

        本项目存在两个不同的数据归一化域:
          - 'hml':  HumanML3D 训练数据使用 dataset/HumanML3D/Mean.npy + Std.npy (mode='train')
          - 'meta': GT评估数据使用 dataset/t2m_mean.npy + t2m_std.npy (mode='gt')

        TMR 使用自己的归一化域: TMR/stats/humanml3d/guoh3dfeats/mean.pt + std.pt
        """
        # 1. 加载 HumanML3D 训练数据的 Mean.npy / Std.npy (mode='train' 使用)
        hml_mean_path = os.path.join(hml_mean_std_dir, "Mean.npy")
        hml_std_path = os.path.join(hml_mean_std_dir, "Std.npy")
        if os.path.exists(hml_mean_path) and os.path.exists(hml_std_path):
            self._hml_mean = torch.from_numpy(np.load(hml_mean_path)).float().to(device)
            self._hml_std = torch.from_numpy(np.load(hml_std_path)).float().to(device)
            print(f"  ✓ Loaded HML stats (train): {hml_mean_path}, {hml_std_path}")
        else:
            raise FileNotFoundError(
                f"HumanML3D stats not found in {hml_mean_std_dir}. "
                f"Expected Mean.npy and Std.npy."
            )

        # 2. 加载 T2M meta 评估数据的 t2m_mean.npy / t2m_std.npy (mode='gt' 使用)
        meta_dir = os.path.dirname(hml_mean_std_dir) if os.path.basename(hml_mean_std_dir) == 'HumanML3D' else 'dataset'
        meta_mean_path = os.path.join(meta_dir, "t2m_mean.npy")
        meta_std_path = os.path.join(meta_dir, "t2m_std.npy")
        if os.path.exists(meta_mean_path) and os.path.exists(meta_std_path):
            self._meta_mean = torch.from_numpy(np.load(meta_mean_path)).float().to(device)
            self._meta_std = torch.from_numpy(np.load(meta_std_path)).float().to(device)
            print(f"  ✓ Loaded meta stats (eval): {meta_mean_path}, {meta_std_path}")
        else:
            # KIT-ML 等数据集可能没有 meta stats，退回到 HML stats
            print(f"  ⚠ Meta stats not found, falling back to HML stats")
            self._meta_mean = self._hml_mean
            self._meta_std = self._hml_std

        # 3. 加载 TMR 的 mean.pt / std.pt
        if tmr_stats_dir is None:
            # 自动推导: ckpt_dir 结构为 TMR/models/models/tmr_humanml3d_guoh3dfeats
            # TMR stats 在 TMR/stats/humanml3d/guoh3dfeats/
            tmr_root = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_dir)))
            tmr_stats_dir = os.path.join(tmr_root, "stats", "humanml3d", "guoh3dfeats")

        tmr_mean_path = os.path.join(tmr_stats_dir, "mean.pt")
        tmr_std_path = os.path.join(tmr_stats_dir, "std.pt")
        if os.path.exists(tmr_mean_path) and os.path.exists(tmr_std_path):
            self.tmr_mean = torch.load(tmr_mean_path, map_location=device, weights_only=True).float()
            self.tmr_std = torch.load(tmr_std_path, map_location=device, weights_only=True).float()
            print(f"  ✓ Loaded TMR stats: {tmr_mean_path}, {tmr_std_path}")
        else:
            raise FileNotFoundError(
                f"TMR stats not found in {tmr_stats_dir}. "
                f"Expected mean.pt and std.pt."
            )

    def set_norm_domain(self, domain: str):
        """切换归一化域: 'hml' (训练数据) 或 'meta' (GT评估数据)"""
        if domain == 'hml':
            self._norm_domain = 'hml'
            self._input_mean = self._hml_mean
            self._input_std = self._hml_std
        elif domain == 'meta':
            self._norm_domain = 'meta'
            self._input_mean = self._meta_mean
            self._input_std = self._meta_std
        else:
            raise ValueError(f"Unknown norm domain: {domain}. Use 'hml' or 'meta'.")
        print(f"  ✓ TMR evaluator norm domain set to: {domain}")

    def _load_pretrained_weights(self, ckpt_dir: str):
        """加载 TMR 预训练权重"""
        weights_dir = os.path.join(ckpt_dir, "last_weights")

        # 加载 motion_encoder
        motion_enc_path = os.path.join(weights_dir, "motion_encoder.pt")
        if os.path.exists(motion_enc_path):
            state_dict = torch.load(motion_enc_path, map_location=self.device)
            self.motion_encoder.load_state_dict(state_dict, strict=True)
            print(f"  ✓ TMR motion_encoder loaded from {motion_enc_path}")
        else:
            raise FileNotFoundError(f"TMR motion_encoder weights not found: {motion_enc_path}")

        # 加载 text_encoder
        text_enc_path = os.path.join(weights_dir, "text_encoder.pt")
        if os.path.exists(text_enc_path):
            state_dict = torch.load(text_enc_path, map_location=self.device)
            self.text_encoder.load_state_dict(state_dict, strict=True)
            print(f"  ✓ TMR text_encoder loaded from {text_enc_path}")
        else:
            raise FileNotFoundError(f"TMR text_encoder weights not found: {text_enc_path}")

        print("  ✓ TMR evaluator weights loaded successfully")

    def encode_text(self, text_list, sample_mean: bool = False):
        """
        编码 raw text → TMR latent space

        Args:
            text_list: List[str] 原始文本列表
            sample_mean: 是否使用均值编码（确定性）

        Returns:
            latent_vec: (B, latent_dim)
            dists: (mu, logvar) 或 None
        """
        # Step 1: DistilBERT → token embeddings (768 dim)
        with torch.no_grad():
            token_output = self.distilbert(text_list)  # {"x": (B, T, 768), "length": (B,)}
            text_embeddings = token_output["x"].to(self.device)
            text_lengths = token_output["length"].to(self.device)

        # Step 2: 构建 mask
        max_len = text_embeddings.shape[1]
        mask = length_to_mask(text_lengths, max_len, device=self.device)

        # Step 3: ACTORStyleEncoder → latent
        latent_vec, dists = self.text_encoder.encode(
            text_embeddings, mask, sample_mean=sample_mean
        )
        return latent_vec, dists

    def _convert_motion_norm(self, motion):
        """
        将 motion 从当前输入归一化域转换到 TMR 归一化域

        输入域 (由 set_norm_domain 控制):
          - 'hml':  motion = (raw - HML_Mean) / HML_Std
          - 'meta': motion = (raw - Meta_Mean) / Meta_Std

        目标域 (TMR): motion_tmr = (raw - TMR_Mean) / TMR_Std

        转换公式: motion_tmr = (motion * input_std + input_mean - tmr_mean) / tmr_std
        """
        # 反归一化到原始域 → 用 TMR 的统计量重新归一化
        raw_motion = motion * self._input_std + self._input_mean
        motion_tmr = (raw_motion - self.tmr_mean) / (self.tmr_std + 1e-12)

        # 调试：首次调用时打印归一化信息
        if not self._debug_norm_printed:
            self._debug_norm_printed = True
            print(f'[TMR Norm Debug] Norm domain: {self._norm_domain}')
            print(f'[TMR Norm Debug] Motion shape: {motion.shape}')
            print(f'[TMR Norm Debug] Input mean (first 5): {self._input_mean[:5].tolist()}')
            print(f'[TMR Norm Debug] Input std  (first 5): {self._input_std[:5].tolist()}')
            print(f'[TMR Norm Debug] TMR mean   (first 5): {self.tmr_mean[:5].tolist()}')
            print(f'[TMR Norm Debug] TMR std    (first 5): {self.tmr_std[:5].tolist()}')
            print(f'[TMR Norm Debug] Before convert: mean={motion.mean().item():.4f}, std={motion.std().item():.4f}')
            print(f'[TMR Norm Debug] After  convert: mean={motion_tmr.mean().item():.4f}, std={motion_tmr.std().item():.4f}')
            # 验证 raw 往返一致性
            raw_from_input = motion * self._input_std + self._input_mean
            raw_from_tmr = motion_tmr * self.tmr_std + self.tmr_mean
            raw_diff = (raw_from_input - raw_from_tmr).abs().max().item()
            print(f'[TMR Norm Debug] Raw roundtrip diff: {raw_diff:.10f}')

        return motion_tmr

    def encode_motion(self, motion, lengths, sample_mean: bool = False):
        """
        编码 motion → TMR latent space

        自动将输入从 HumanML3D 归一化域转换到 TMR 归一化域。

        Args:
            motion: (B, T, 263) HumanML3D motion features (全部 263 维，含 foot contact)
            lengths: (B,) 每帧实际长度
            sample_mean: 是否使用均值编码（确定性）

        Returns:
            fid_emb: (B, latent_dim) 用于 FID 计算的 embedding（与 return_vecs 相同）
            return_vecs: (B, latent_dim) 潜在向量
            dists: (mu, logvar) 或 None — 分布参数
        """
        motion = motion.to(self.device).float()
        lengths = lengths.to(self.device)

        # 归一化域转换: HumanML3D → TMR
        motion = self._convert_motion_norm(motion)

        max_len = motion.shape[1]
        mask = length_to_mask(lengths, max_len, device=self.device)
        latent_vec, dists = self.motion_encoder.encode(
            motion, mask, sample_mean=sample_mean
        )
        # 返回 3 个值，兼容 eval_cmc.py 的调用接口: fid_emb, return_vecs, dists
        return latent_vec, latent_vec, dists

    def get_motion_embeddings(self, motions, m_lens):
        """获取 motion embeddings（无梯度，兼容 GRU evaluator 接口）"""
        with torch.no_grad():
            _, motion_emb, _ = self.encode_motion(motions, m_lens, sample_mean=True)
        return motion_emb

    def get_motion_embeddings_with_grad(self, motions, m_lens):
        """获取 motion embeddings（带梯度，用于训练时的损失计算）"""
        with torch.enable_grad():
            motions = motions.to(self.device).float()
            m_lens = m_lens.to(self.device)
            _, motion_emb, _ = self.encode_motion(motions, m_lens, sample_mean=False)
        return motion_emb

    def get_co_embeddings_with_grad(self, text_list, motions, m_lens):
        """
        同时获取 text 和 motion embeddings（带梯度）

        注意: 与 GRU evaluator 接口不同，TMR 版本接收 raw text strings 而非 word_embeddings/pos_ohot/cap_lens

        Args:
            text_list: List[str] 原始文本
            motions: (B, T, 263) motion features
            m_lens: (B,) motion lengths

        Returns:
            text_emb: (B, latent_dim)
            motion_emb: (B, latent_dim)
        """
        with torch.enable_grad():
            motions = motions.to(self.device).float()
            m_lens = m_lens.to(self.device)

            # 编码 motion (3 返回值: fid_emb, latent_vec, dists)
            _, motion_emb, _ = self.encode_motion(motions, m_lens, sample_mean=False)

            # 编码 text (DistilBERT 部分冻结，text_encoder 可过梯度)
            text_emb, _ = self.encode_text(text_list, sample_mean=False)

        return text_emb, motion_emb

    def get_co_embeddings(self, text_list=None, motions=None, m_lens=None,
                          word_embs=None, pos_ohot=None, cap_lens=None, captions=None):
        """
        获取 text 和 motion embeddings（无梯度）

        兼容两种调用方式:
        1. TMR 直接调用: get_co_embeddings(text_list, motions, m_lens)
        2. eval_cmc.py keyword 调用: get_co_embeddings(word_embs=..., pos_ohot=..., cap_lens=..., motions=..., m_lens=..., captions=...)
           此时使用 captions (raw text) 来编码文本
        """
        # 确定 text 输入: 优先使用 text_list，其次 captions
        if text_list is None:
            text_list = captions
        # 如果 key 参数传递了 motions/m_lens，优先使用
        if motions is None:
            raise ValueError("motions must be provided to get_co_embeddings")

        with torch.no_grad():
            text_emb, _ = self.encode_text(text_list, sample_mean=True)
            _, motion_emb, _ = self.encode_motion(motions, m_lens, sample_mean=True)
        return text_emb, motion_emb

    def train(self, mode: bool = True) -> nn.Module:
        """
        设置训练/评估模式
        - motion_encoder / text_encoder: 允许梯度通过（用于训练扩散模型）
        - distilbert: 始终冻结
        """
        super().train(mode)
        self.motion_encoder.train(mode)
        self.text_encoder.train(mode)
        # distilbert 始终 eval
        self.distilbert.eval()
        return self

    def eval(self) -> nn.Module:
        """设置评估模式"""
        return self.train(False)
