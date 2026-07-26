import torch.nn as nn
from .transformer import STTransformerLayer

class SwinTrans(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        
        # transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(opt)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range((self.opt.n_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(opt))
            self.output_blocks.append(STTransformerLayer(opt))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))