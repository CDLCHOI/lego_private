# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.scale = scale
        print('--- creating CFG  1model')
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        if self.scale == 1:
            return self.model(x, timesteps, y)

        B = x.shape[0]
        cond_mode = self.model.cond_mode
        # assert cond_mode in ['only_text', 'only_spatial', 'both_text_spatial','text']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        out = self.model(x, timesteps, y)
        if cond_mode == 'no_cond':
            assert torch.allclose(out, out_uncond)
        scale_tensor = torch.ones((B), device=x.device).view(-1, 1, 1, 1) * self.scale
        return out_uncond + (scale_tensor * (out - out_uncond))

class CFG_SALAD(nn.Module):

    def __init__(self, model, scale=7.5):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.scale = scale
        print('--- creating CFG_SALAD')
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        if self.scale == 1:
            return self.model(x, timesteps, y['text'])

        B = x.shape[0]
        none_text = [""] * B

        out_uncond = self.model(x, timesteps, none_text)
        out = self.model(x, timesteps, y['text'])
        scale_tensor = torch.ones((B), device=x.device).view(-1, 1, 1) * self.scale
        return out_uncond + (scale_tensor * (out - out_uncond))
    
class CFG3(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run
        print('--- creating CFG3Model')
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        xt_all_noisy = y['xt_all_noisy']
        scale = y['scale'].view(-1, 1, 1, 1)
        # assert cond_mode in ['only_text', 'only_spatial', 'both_text_spatial','text']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        # 1
        out1 = self.model(xt_all_noisy, timesteps, y_uncond) 
        # 2
        out2 = self.model(x, timesteps, y) 
        '''
        无条件输出 + 所有条件的联合偏移量
        '''
        return out1 + scale * (out2 - out1)