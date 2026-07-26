"""
This code was inspired by the denoiser implementation in the Motion Latent Diffusion
    - https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/architectures/mld_denoiser.py
"""

from typing import List
import torch
import torch.nn as nn
import numpy as np

from models.salad.denoiser.clip import FrozenCLIPTextEncoder
from models.salad.denoiser.embedding import TimestepEmbedding, PositionalEmbedding
from models.salad.denoiser.transformer import SkipTransformer
from models.salad.vae.encdec import MotionEncoder, MotionDecoder, STConvEncoder, STConvDecoder
from data_loaders.humanml.utils.get_opt import get_opt



class SALAD2(nn.Module):
    def __init__(self):
        super().__init__()

        opt = get_opt('/home/deli/project/salad/checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt', None)
        vae_opt = get_opt('/home/deli/project/salad/checkpoints/t2m/t2m_vae_gelu/opt.txt', None)

        self.latent_dim = opt.latent_dim
        self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768 # ViT-L/14
        self.cond_mask_prob = 0.1

        #### 编码器
        self.motion_enc = MotionEncoder(vae_opt)
        self.motion_dec = MotionDecoder(vae_opt)
        
        # timestep embedding
        self.timestep_emb = TimestepEmbedding(self.latent_dim)

        # CLIP text encoder
        self.clip_model = FrozenCLIPTextEncoder(opt)
        self.word_emb = nn.Linear(self.clip_dim, self.latent_dim)
        
        # positional embedding
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)

        # transformer
        self.transformer = SkipTransformer(opt)
        self.swin_trans = nn.ModuleList()


        # cache for CLIP embedding
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None
    
    
    def remove_clip_cache(self):
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

    def forward(self, x_input, timestep, text, len_mask=None):
        ''' 
        x_input: (b,196,263)
        len_mask: (b,196), 原代码是(b,49)因为有4倍下采样  len_mask = lengths_to_mask(m_lens) 
        '''
        x = self.motion_enc(x_input)
        text = ["" if np.random.rand(1) < self.cond_mask_prob else t for t in text]
        z_out = self.latent_denoise(x, timestep, text, len_mask=len_mask)
        x_output = self.motion_dec(x)

        return x_output


    def latent_denoise(self, x, timestep_emb, text, len_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, use_cached_clip=False):
        """
        sample: [B, T, J, D]
        timestep: [B,] or [1,]
        lengths: [B,]
        """

        # input process
        B, T, J, D = x.size()

        # diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # text embedding
        if use_cached_clip and all([e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]]):
            word_emb = self._cache_word_emb
            ca_mask = self._cache_ca_mask
            token_pos = self._cache_tokens_pos
        else:
            word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)
            word_emb = self.word_emb(word_emb)
            # if use_cached_clip:
            #     self._cache_word_emb = word_emb
            #     self._cache_ca_mask = ca_mask
            #     self._cache_tokens_pos = token_pos
        
        # positional embedding
        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D) # (b,196,22,dim)

        # attention masks
        if len_mask is not None:
            len_mask = len_mask.repeat_interleave(J, dim=0)

        # transformer
        x, attn_weights = self.transformer.forward(x, timestep_emb, word_emb,
                                                   sa_mask=None if len_mask is None else ~len_mask,
                                                   ca_mask=~ca_mask,
                                                   need_attn=need_attn,
                                                   fixed_sa=fixed_sa,
                                                   fixed_ta=fixed_ta,
                                                   fixed_ca=fixed_ca)
        
        # (b,196,22,dim)
        # (b,98,22,2dim)
        # (b,49,22,4dim)
        # (b,98,22,2dim)
        # (b,196,22,dim)


        return x