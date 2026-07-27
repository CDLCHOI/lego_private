"""测试 T5TextEncoder 是否能成功创建"""
import torch
import sys
sys.path.insert(0, '/home/deli/project/reward_mdm')

# 先测试最基础的 tokenizer 加载
print("=" * 60)
print("Test 1: 直接加载 AutoTokenizer")
print("=" * 60)

from transformers import AutoTokenizer, T5EncoderModel

model_name = 'google/t5-v1_1-base'

# 测试1: use_fast=False
print(f"\n>>> 尝试 use_fast=False ...")
try:
    tokenizer_slow = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        legacy=False,
        use_fast=False,
    )
    print("✅ use_fast=False 成功!")
except Exception as e:
    print(f"❌ use_fast=False 失败: {e}")

# 测试2: use_fast=True (默认行为)
print(f"\n>>> 尝试 use_fast=True (默认) ...")
try:
    tokenizer_fast = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        legacy=False,
        use_fast=True,
    )
    print("✅ use_fast=True 成功!")
except Exception as e:
    print(f"❌ use_fast=True 失败: {e}")

# 测试3: 使用 T5Tokenizer 直接加载
print(f"\n>>> 尝试 T5Tokenizer.from_pretrained ...")
try:
    from transformers import T5Tokenizer
    tokenizer_t5 = T5Tokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        legacy=False,
    )
    print("✅ T5Tokenizer.from_pretrained 成功!")
except Exception as e:
    print(f"❌ T5Tokenizer.from_pretrained 失败: {e}")

# 测试4: 使用 T5TokenizerFast 直接加载
print(f"\n>>> 尝试 T5TokenizerFast.from_pretrained ...")
try:
    from transformers import T5TokenizerFast
    tokenizer_t5_fast = T5TokenizerFast.from_pretrained(
        model_name,
        local_files_only=False,
    )
    print("✅ T5TokenizerFast.from_pretrained 成功!")
except Exception as e:
    print(f"❌ T5TokenizerFast.from_pretrained 失败: {e}")

# 测试5: 完整 T5TextEncoder
print("\n" + "=" * 60)
print("Test 2: 完整的 T5TextEncoder")
print("=" * 60)

from models.snapmogen_evaluator.encode_text import T5TextEncoder

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    encoder = T5TextEncoder(
        device=device,
        local_files_only=False,
        from_pretrained=model_name,
        model_max_length=120,
    )
    print("✅ T5TextEncoder 创建成功!")

    # 测试 encode
    texts = ["a person is walking forward", "someone jumps high"]
    emb, mask = encoder.get_text_embeddings(texts)
    print(f"   Text embeddings shape: {emb.shape}")
    print(f"   Mask shape: {mask.shape}")
    print("✅ T5TextEncoder 推理成功!")
except Exception as e:
    print(f"❌ T5TextEncoder 创建失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 完整 EvaluatorWrapper
print("\n" + "=" * 60)
print("Test 3: 完整的 EvaluatorWrapper")
print("=" * 60)

from utils.config_utils import load_config
from models.snapmogen_evaluator.evaluator_wrapper import EvaluatorWrapper

try:
    eval_cfg = load_config('./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml')
    print(f"dim_pose: {eval_cfg.data.dim_pose}")
    print(f"text_embedder.version: {eval_cfg.text_embedder.version}")

    wrapper = EvaluatorWrapper(
        eval_cfg,
        device=device,
        model_path='./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/model/net_best_top1.tar',
    )
    print("✅ EvaluatorWrapper 创建成功!")

    # 测试编码
    wrapper.eval()
    text_emb, dist = wrapper.encode_text(["a person walks forward", "jumping jacks exercise"])
    print(f"   Text embedding shape: {text_emb.shape}")

    # 测试 motion 编码
    motion = torch.randn(2, 320, 148).to(device)
    lengths = torch.tensor([320, 200]).to(device)
    fid_emb, motion_emb, dist_m = wrapper.encode_motion(motion, lengths)
    print(f"   Motion embedding shape: {motion_emb.shape}")
    print("✅ EvaluatorWrapper 推理成功!")
except Exception as e:
    print(f"❌ EvaluatorWrapper 创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("全部测试完成")
print("=" * 60)
