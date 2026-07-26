import torch
import clip
import loratorch as lora
from utils.lora_util import apply_lora_attn_mlp
import argparse
import os


def merge_lora_weights(clip_version, lora_checkpoint_path, output_path):
    """
    合并LoRA权重到原始CLIP模型中
    
    Args:
        clip_version: CLIP模型版本，如'ViT-B/32'
        lora_checkpoint_path: 包含训练好的LoRA权重的检查点路径
        output_path: 合并后的模型保存路径
    """
    # 1. 加载原始CLIP模型
    print(f"Loading original CLIP model: {clip_version}")
    original_clip, _ = clip.load(clip_version, device='cpu', jit=False)
    
    # 2. 为CLIP模型应用LoRA层
    print("Applying LoRA layers to CLIP model")
    lora_clip = apply_lora_attn_mlp(original_clip, encoder_type='text', mlp=True, attn=True)
    
    # 3. 加载训练好的LoRA权重
    print(f"Loading LoRA weights from: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')
    
    if 'clip_lora' in checkpoint:
        lora_weights = checkpoint['clip_lora']
    else:
        # 如果直接保存了整个模型的权重
        lora_weights = checkpoint
    
    # 4. 加载LoRA权重到模型
    # 处理权重键名，移除可能的前缀
    processed_weights = {}
    for key, value in lora_weights.items():
        if key.startswith('clip_model.'):
            processed_key = key[11:]  # 移除'clip_model.'前缀
        else:
            processed_key = key
        processed_weights[processed_key] = value
    
    missing_keys, unexpected_keys = lora_clip.load_state_dict(processed_weights, strict=False)
    
    # 5. 创建新的原始结构CLIP模型用于保存合并后的权重
    merged_clip, _ = clip.load(clip_version, device='cpu', jit=False)
    
    # 6. 合并权重
    print("Merging LoRA weights with original CLIP weights")
    
    # 遍历所有参数，将LoRA权重合并到原始权重中
    with torch.no_grad():
        # 处理text encoder的resblocks
        for i, (lora_resblock, orig_resblock) in enumerate(zip(lora_clip.transformer.resblocks, merged_clip.transformer.resblocks)):
            
            # 合并注意力层权重
            # 包含的key: 'in_proj_weight', 'in_proj_bias', 'o_lora_A', 'o_lora_B', 'qkv_lora_A', 'qkv_lora_B', 'out_proj.weight', 'out_proj.bias'
            if hasattr(lora_resblock.attn, 'in_proj_weight') and hasattr(lora_resblock.attn, 'qkv_lora_A'):
                # 获取LoRA缩放因子
                scaling = lora_resblock.attn.lora_alpha / lora_resblock.attn.r
                
                # 合并QKV权重
                if hasattr(lora_resblock.attn, 'qkv_lora_A') and hasattr(lora_resblock.attn, 'qkv_lora_B'):
                    qkv_lora_A = lora_resblock.attn.qkv_lora_A
                    qkv_lora_B = lora_resblock.attn.qkv_lora_B
                    
                    # 计算LoRA贡献并合并到in_proj_weight
                    orig_resblock.attn.in_proj_weight += (qkv_lora_B @ qkv_lora_A) * scaling
                
                # 合并O权重
                if hasattr(lora_resblock.attn, 'o_lora_A') and hasattr(lora_resblock.attn, 'o_lora_B'):
                    o_lora_A = lora_resblock.attn.o_lora_A
                    o_lora_B = lora_resblock.attn.o_lora_B
                    
                    # 合并到out_proj.weight
                    orig_resblock.attn.out_proj.weight += (o_lora_B @ o_lora_A) * scaling
            
            # 合并MLP层权重
            # mlp.c_fc层包含的key: 'c_fc.weight', 'c_fc.bias', 'c_fc.w_lora_A', 'c_fc.w_lora_B'
            if hasattr(lora_resblock.mlp.c_fc, 'w_lora_A') and hasattr(lora_resblock.mlp.c_fc, 'w_lora_B'):
                scaling = lora_resblock.mlp.c_fc.lora_alpha / lora_resblock.mlp.c_fc.r
                lora_A = lora_resblock.mlp.c_fc.w_lora_A
                lora_B = lora_resblock.mlp.c_fc.w_lora_B
                
                # 合并权重
                orig_resblock.mlp.c_fc.weight += (lora_B @ lora_A) * scaling
                
            
            # mlp.c_proj包含的key: 'c_proj.weight', 'c_proj.bias', 'c_proj.w_lora_A', 'c_proj.w_lora_B'
            if hasattr(lora_resblock.mlp.c_proj, 'w_lora_A') and hasattr(lora_resblock.mlp.c_proj, 'w_lora_B'):
                scaling = lora_resblock.mlp.c_proj.lora_alpha / lora_resblock.mlp.c_proj.r
                lora_A = lora_resblock.mlp.c_proj.w_lora_A
                lora_B = lora_resblock.mlp.c_proj.w_lora_B
                
                # 合并权重
                orig_resblock.mlp.c_proj.weight += (lora_B @ lora_A) * scaling
                
    
    # 8. 验证合并前后的模型输出是否一致
    print("\nVerifying merged model...")
    
    # 设置两个模型为评估模式
    lora_clip.eval()
    merged_clip.eval()
    
    # 创建随机输入tensor（模拟text encoder的输入）
    batch_size = 2
    seq_length = 77  # CLIP文本输入的最大长度
    
    # 创建随机token索引
    random_tokens = torch.randint(0, 49408, (batch_size, seq_length))  # 49408是CLIP词汇表大小
    
    with torch.no_grad():
        # 获取合并前的LoRA模型输出
        lora_output = lora_clip.encode_text(random_tokens)
        
        # 获取合并后的模型输出
        merged_output = merged_clip.encode_text(random_tokens)
    
    # 打印输出信息
    print("\nOutput from LoRA CLIP (before merge):")
    print(lora_output.flatten()[:5])
    print("\nOutput from merged CLIP (after merge):")
    print(merged_output.flatten()[:5])
    
    # 使用torch.allclose验证输出是否一致
    tolerance = 1e-5  # 容差
    is_close = torch.allclose(lora_output, merged_output, rtol=tolerance, atol=tolerance)
    
    print(f"\nVerification result: {'PASSED' if is_close else 'FAILED'}")
    print(f"All close with rtol={tolerance}, atol={tolerance}: {is_close}")
    
    if not is_close:
        # 计算差异
        diff = torch.abs(lora_output - merged_output)
        print(f"\nMax difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")
    
    # 7. 保存合并后的模型
    print(f"\nSaving merged model to: {output_path}")
    torch.save(merged_clip.state_dict(), output_path)
    
    print("LoRA weights merged successfully!")
    print(f"Merged model saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into original CLIP model")
    parser.add_argument("--clip_version", type=str, default="ViT-B/32", help="CLIP model version")
    parser.add_argument("--lora_checkpoint_path", type=str, help="Path to LoRA checkpoint file")
    parser.add_argument("--output_path", type=str, default="merged_clip.pt", help="Output path for merged model")
    
    args = parser.parse_args()

    args.lora_checkpoint_path = 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ricglobal1/net_best.pth'
    args.output_path = args.lora_checkpoint_path.replace(os.path.basename(args.lora_checkpoint_path), 'merged_clip.pth')
    if os.path.exists(args.output_path):
        raise FileExistsError(f"Output file {args.output_path} already exists. ")
    
    merge_lora_weights(args.clip_version, args.lora_checkpoint_path, args.output_path)