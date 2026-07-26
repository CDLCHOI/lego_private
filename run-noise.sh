
# 原MDM 无el50
python train.py --exp_name 0619_mdm_step50_noisyonce5 --batch_size 128 --gpu 0 --overwrite --print_iter 200 --save_iter 10000 --total_iter 600000 --lr 1e-4 --lr-scheduler 500000 --modeltype mdm --eval_during_train --eval_sample_num 10000 --diffusion_steps 50 --init_noisy_data_level 5
# 用el50
python train.py --exp_name 0619_mdm_step50_noisyonce5_el50 --batch_size 128 --gpu 2 --overwrite --print_iter 200 --save_iter 10000 --total_iter 600000 --lr 1e-4 --lr-scheduler 500000 --modeltype mdm --eval_during_train --eval_sample_num 10000 --diffusion_steps 50 --init_noisy_data_level 5 --emb_loss 50

python eval_cmc.py --modeltype mdm_bert --replication_times 10 --gpu 4 --resume_trans output/0619_mdm_step50_noisyonce5_el50/net_best.pth --diffusion_steps 50


# 用噪声评估器训练，原评估器测试
python train.py --exp_name 0624_mdm_el50_noiseeval --batch_size 128 --gpu 6 --overwrite --print_iter 100 --save_iter 10000 --total_iter 400000 --lr 1e-4 --lr-scheduler 200000 --gamma 0.1 --modeltype mdm --emb_loss 50 --eval_during_train --diffusion_steps 50 --resume_trans output/humanml_enc_512_50steps/model000750000.pt --eval_sample_num 10000 --evaluator_train /home/deli/project/text-to-motion/checkpoints/t2m/0608_evaluator_union_noisy010/model/finest.tar