
默认参数
--batch_size 64 --total_iter 600000 --save_iter 5000 --print_iter 100 --lr 1e-4 --gamma 0.1 --lr-scheduler 300000 --diffusion_steps 50 

################ HumanML3D 训练
# 原MDM
python train.py --exp_name 0814_MDMCLIP --batch_size 64 --gpu 4 --overwrite --print_iter 100 --save_iter 5000 --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 --gamma 0.1 --modeltype mdm_bert --text_encoder_type clip --eval_during_train --diffusion_steps 50
python train.py --exp_name 0814_MDMCLIPlora_cl10_tcl1_0716_scratch --overwrite --gpu 1 --modeltype mdm_bert --text_encoder_type clip --cos_loss 10 --text_cos_loss 1 --eval_during_train --evaluator_train /home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar --add_clip_lora

# 消融实验 采用评估器的text_encoder试试效果
python train.py --exp_name 0826_MDMGRU --batch_size 64 --gpu 3 --overwrite --print_iter 100 --save_iter 5000 --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 --gamma 0.1 --modeltype mdm_bert --text_encoder_type gru --eval_during_train --diffusion_steps 50
python train.py --exp_name 0826_MDMGRU_cl10_tcl1_0716 --batch_size 64 --gpu 6 --overwrite --print_iter 100 --save_iter 5000 --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 --gamma 0.1 --modeltype mdm_bert --text_encoder_type gru --eval_during_train --diffusion_steps 50 --cos_loss 10 --text_cos_loss 1 --evaluator_train /home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar
# 消融 高级loss只更新CLIP，MSE loss只更新MDM
python train.py --exp_name 0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ablation_separate_update --batch_size 64 --gpu 7 --overwrite --modeltype mdm_bert --text_encoder_type clip --cos_loss 10 --text_cos_loss 2 --eval_during_train --diffusion_steps 50 --evaluator_train /home/deli/project/text-to-motion/checkpoints/t2m/0716_evaluator32_infosim_fixmovement_cos5/model/finest.tar --add_clip_lora --ablation_separate_update



################ HumanML3D 测试 --add_clip_lora 后期要改为 --lora_clip
python eval_lora_mdm.py --modeltype mdm_bert --text_encoder_type clip --add_clip_lora --replication_times 10 --gpu 7 --resume_trans output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/net_best.pth --diffusion_steps 50 --eval_mode with_mm


################ KIT 训练
# 使用ml3d训练好的lora_clip，训练kit，仅用mse直接训。应该要比常规训练kit要好
python train.py --exp_name 0814_KIT_MDMCLIP_preatrainlora_scratch --dataset_name kit --overwrite --gpu 7 --modeltype mdm_bert --text_encoder_type clip --eval_during_train --add_clip_lora --pretrained_lora_path output/0814_MDMCLIPlora_cl10_tcl1_0716_scratch/net_best.pth
# 对比实验，常规训练kit。
python train.py --exp_name 0814_KIT_MDMCLIP_scratch --dataset_name kit --overwrite --gpu 1 --modeltype mdm_bert --text_encoder_type clip --eval_during_train
# 用info cos的kit评估器调lora_clip+ scratch MDM
python train.py --exp_name 0820_KIT_MDMCLIPlora_cl10_tcl1_0820_scratch --dataset_name kit --overwrite --gpu 6 --modeltype mdm_bert --text_encoder_type clip --cos_loss 10 --text_cos_loss 1 --eval_during_train --evaluator_train /home/deli/project/text-to-motion/checkpoints/kit/0820_kit_evaluator32_info_cos1/model/finest.tar --add_clip_lora
# 用info cos的kit评估器调lora_clip+ 预训练 MDM
python train.py --exp_name 0820_KIT_MDMCLIPlora_cl10_tcl1_0820 --dataset_name kit --overwrite --gpu 5 --modeltype mdm_bert --text_encoder_type clip --cos_loss 10 --text_cos_loss 1 --eval_during_train --evaluator_train /home/deli/project/text-to-motion/checkpoints/kit/0820_kit_evaluator32_info_cos1/model/finest.tar --add_clip_lora --resume_trans output/0820_KIT_MDMCLIP_scratch/net_best.pth --batch_size 32 --total_iter 300000 --save_iter 2000 --print_iter 100 --lr 1e-4 --gamma 0.1 --lr-scheduler 150000

# KIT 验证
python eval_lora_mdm.py --dataset_name kit --modeltype mdm_bert --text_encoder_type clip --add_clip_lora --replication_times 10 --gpu 2 --resume_trans output/0821_KIT_MDMCLIPlora_cl10_tcl1_0820_scratch/net_best.pth --diffusion_steps 50


