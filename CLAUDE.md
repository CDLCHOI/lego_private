# CRITICAL RULE: ALWAYS ANSWER IN SIMPLIFIED CHINESE.
# 核心指令：无论用户用何种语言提问，所有回复、思考、解释必须一律使用简体中文！
# 同样功能的代码能够抽象成为函数一定要单独抽象成为函数
# 严禁删除任何文件！如果需要删除，你告诉我，我自己去删除。
# 有任何不清楚的地方，停下来问我，征求我的意见，不要在模糊不清的时候强行执行。
# 本仓库实现的功能是text to motion generation。
# 主要网络（我称作LEGO）的训练代码是：python train.py --modeltype mdm_bert --cos_loss 10 --text_cos_loss 2 --evaluator_train /home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar --add_clip_lora。
## 几个关键参数的解释
1) --modeltype mdm_bert 采用Transformer decoder作为网络结构
2) --evaluator_train 的作用是读取一个evaluator，这个evaluator实际上就是作为projector，将text和motion给映射到latent vecto
3) --add_clip_lora的作用是给CLIP text encoder添加lora的可学习参数，目视是为了训练diffusion model的时候同时微调CLIP text encode
4) --cos_loss 和 --text_cos_loss 是gt_motion和pred_motion的余弦相似度损失函数的系数，以及pred_motion和text的余弦相似度损失函数的系数。

