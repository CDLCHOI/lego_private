"""
TMR 核心网络结构模块
从 TMR 工程迁移，不依赖 TMR 的外部软连接
包含: PositionalEncoding, ACTORStyleEncoder, ACTORStyleDecoder
"""
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from einops import repeat


def length_to_mask(length, max_len, device: torch.device = None) -> Tensor:
    """将序列长度转换为 mask"""
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)
    elif isinstance(length, torch.Tensor):
        length = length.to(device)

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    """Transformer 位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class ACTORStyleEncoder(nn.Module):
    """
    ACTOR 风格的 Transformer Encoder
    用于 motion encoder 和 text encoder (TMR 中两者结构相同，仅 nfeats 不同)

    - motion encoder: nfeats=263 (guoh3dfeats, 包含 4 维 foot contact)
    - text encoder: nfeats=768 (DistilBERT token embeddings)
    """
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.vae = vae

        self.projection = nn.Linear(nfeats, latent_dim)

        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, x_dict: Dict) -> Tensor:
        """
        Args:
            x_dict: {"x": (B, T, nfeats), "mask": (B, T) bool}
        Returns:
            (B, nbtokens, latent_dim) — VAE 时 nbtokens=2 (mu, logvar), 否则 nbtokens=1
        """
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]

    def encode(self, x: Tensor, mask: Tensor, sample_mean: bool = False):
        """
        Encode input to latent vector

        Args:
            x: (B, T, nfeats) input features
            mask: (B, T) bool mask (True = valid)
            sample_mean: 是否采样均值 (确定性编码)

        Returns:
            latent_vec: (B, latent_dim) 潜在向量
            dists: (mu, logvar) 分布参数，vae=False 时返回 None
        """
        output = self.forward({"x": x, "mask": mask})

        if self.vae:
            dists = output.unbind(1)  # (mu, logvar) — each (B, latent_dim)
            mu, logvar = dists
            logvar = torch.clamp(logvar, -10.0, 10.0)
            if sample_mean:
                latent_vec = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vec = mu + eps * std
        else:
            (latent_vec,) = output.unbind(1)
            dists = None

        return latent_vec, dists
