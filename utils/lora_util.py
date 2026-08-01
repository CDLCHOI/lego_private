import loratorch as lora
import torch

def apply_lora_attn_mlp(model, encoder_type="text", rank=16, lora_alpha=32, mlp=True, attn=True):
    if encoder_type == 'visual':
        encoder = model.visual.transformer
    elif encoder_type == 'text':
        encoder = model.transformer
    else:
        raise ValueError("Invalid encoder_type. Choose 'visual' or 'text'.")

    enable_lora=['q', 'k', 'v', 'o']
    for i, resblock in enumerate(encoder.resblocks):
        if hasattr(resblock, 'attn') and attn:
            multihead = resblock.attn
            lora_multihead = lora.MultiheadAttention(r=rank,
                                    lora_alpha=lora_alpha,
                                    enable_lora=enable_lora,
                                    embed_dim=multihead.embed_dim,
                                    num_heads=multihead.num_heads,
                                    dropout=multihead.dropout,
                                    bias=True if hasattr(multihead, "in_proj_bias") else False,
                                    add_bias_kv=False if multihead.bias_k==None else True,
                                    add_zero_attn=multihead.add_zero_attn,
                                    kdim=multihead.kdim,
                                    vdim=multihead.vdim,
                                    batch_first=multihead.batch_first)
            missing_keys, unexpected_keys = lora_multihead.load_state_dict(multihead.state_dict(), strict=False)
            resblock.attn = lora_multihead

        if hasattr(resblock, 'mlp') and mlp:
            old_mlp_fc=resblock.mlp.c_fc
            old_mlp_proj=resblock.mlp.c_proj
            new_mlp_fc = lora.Linear(
                old_mlp_fc.in_features,
                old_mlp_fc.out_features,
                bias=True if hasattr(old_mlp_fc, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            new_mlp_proj = lora.Linear(
                old_mlp_proj.in_features,
                old_mlp_proj.out_features,
                bias=True if hasattr(old_mlp_proj, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            c, d = new_mlp_fc.load_state_dict(old_mlp_fc.state_dict(),strict=False)
            e,f = new_mlp_proj.load_state_dict(old_mlp_proj.state_dict(),strict=False)
            resblock.mlp.c_fc = new_mlp_fc
            resblock.mlp.c_proj = new_mlp_proj

    lora.mark_only_lora_as_trainable(model)
    return model

# def load_and_freeze_lora_weights(model, clip_lora_path='/home/deli/project/reward_mdm/output/0811_MDMCLIPlora_cl10_tcl1_0716/net_best_diff.pth'):
#     for name, param in model.clip_model.named_parameters():
#         if "lora" in name:   
#             param.requires_grad = False

#     lora_dict = torch.load(clip_lora_path)['clip_lora']
#     print(f' === loading clip_lora dict from {clip_lora_path}')
#     lora_dict_new = {}
#     for k, v in lora_dict.items():
#         kk = k.replace('clip_model.', '')
#         lora_dict_new[kk] = v
#     model.clip_model.load_state_dict(lora_dict_new, strict=False)

def apply_lora_attn_mlp_bert(model, rank=16, lora_alpha=32, mlp=True, attn=True):
    """对 DistilBERT 文本编码器的 attention 和/或 FFN 层添加 LoRA 可学习参数。

    DistilBERT 的 transformer 层结构：
        layer.attention: q_lin, k_lin, v_lin, out_lin (nn.Linear)
        layer.ffn:       lin1, lin2 (nn.Linear)

    Args:
        model: BERT wrapper (models/mdm_bert/BERT_encoder.py 中的 BERT 实例)
        rank: LoRA 秩
        lora_alpha: LoRA alpha 系数
        mlp: 是否对 FFN 层添加 LoRA
        attn: 是否对 attention 层添加 LoRA
    Returns:
        model: 添加了 LoRA 层的模型
    """
    for layer in model.text_model.transformer.layer:
        if attn:
            attention = layer.attention
            for lin_name in ['q_lin', 'k_lin', 'v_lin', 'out_lin']:
                old_linear = getattr(attention, lin_name)
                new_linear = lora.Linear(
                    old_linear.in_features,
                    old_linear.out_features,
                    bias=(old_linear.bias is not None),
                    r=rank,
                    lora_alpha=lora_alpha,
                )
                new_linear.load_state_dict(old_linear.state_dict(), strict=False)
                setattr(attention, lin_name, new_linear)

        if mlp:
            ffn = layer.ffn
            for lin_name in ['lin1', 'lin2']:
                old_linear = getattr(ffn, lin_name)
                new_linear = lora.Linear(
                    old_linear.in_features,
                    old_linear.out_features,
                    bias=(old_linear.bias is not None),
                    r=rank,
                    lora_alpha=lora_alpha,
                )
                new_linear.load_state_dict(old_linear.state_dict(), strict=False)
                setattr(ffn, lin_name, new_linear)

    lora.mark_only_lora_as_trainable(model)
    return model


def load_lora_mdm_for_eval(net, ckpt_path, use_lora=True, logger=None):
    assert use_lora, 'use_lora must be True'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    assert 'base' in ckpt.keys() and 'clip_lora' in ckpt.keys(), f'{str(ckpt.keys())}'
    # === Load MDM weights
    mdm_weights = ckpt['base']
    missing_keys, unexpected_keys = net.load_state_dict(mdm_weights, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

    # === Load LORA weights
    if use_lora:
        lora_weights = ckpt['clip_lora']
        net.load_state_dict(lora_weights, strict=False)
        if logger is not None:
            logger.info(f'load lora ckpt from {ckpt_path}')

    # === Freeze LORA weights
    for name, param in net.clip_model.named_parameters():
        if "lora" in name:   
            param.requires_grad = False
