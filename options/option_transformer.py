import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_no_vald', action='store_true', default=False)
    parser.add_argument('--train_with_test_set', action='store_true', default=False)
    parser.add_argument('--sample_mean', action='store_true', default=False, help='评估时 encode_motion 使用 sample_mean=True（确定性 mu 而非随机采样 z）')
    parser.add_argument('--evaluator_eval_type', type=str, default=None, choices=['gru', 'snapmogen', 'tmr'])
    parser.add_argument('--evaluator_train_type', type=str, default='gru', choices=['gru', 'snapmogen', 'tmr'])
    parser.add_argument('--evaluator_train_dim_pose', type=int, default=None, choices=[148,292], help='SnapMoGen evaluator motion encoder 的输入维度（148 或 292）')
    parser.add_argument('--correct_snapmogen_norm', action='store_true', default=False, help='使用从训练集重新计算的 mean/std（dataset/snapmogen_norm/）替代官方预置的错误归一化参数')
    parser.add_argument('--correct_snapmogen_norm_all', action='store_true', default=False, help='使用从全数据集（train+test）计算的逐元素 mean/std（dataset/snapmogen_norm/mean_all.npy, std_all.npy）。优先级高于 --correct_snapmogen_norm')
    parser.add_argument('--snapmogen_no_norm', action='store_true', default=False, help='不进行任何归一化，直接将原始 motion 数据送入扩散模型训练（mean=0, std=1）')
    parser.add_argument('--random_gt', action='store_true', default=False)
    parser.add_argument('--ablation_separate_update', action='store_true', default=False) # 高级loss只更新CLIP，MSE loss只更新MDM
    parser.add_argument('--train_sample_num', type=int, default=0)
    parser.add_argument('--eval_mode', type=str, default='no_mm', choices=['no_mm', 'with_mm'])
    parser.add_argument('--pretrained_lora_path', type=str, default=None) # 使用ml3d训练的lora并冻结，去训练kit数据集
    parser.add_argument('--add_clip_lora', action='store_true', default=False) # 是否训CLIP
    parser.add_argument('--random_text_encoder', action='store_true', default=False) # 无视文本编码器预训练权重，采用随机初始化
    parser.add_argument('--text_encoder_type', type=str, default='bert', choices=['bert', 'clip', 'gru', 'lamp'])
    parser.add_argument('--no_random', action='store_true', default=False) # dataset 关闭随机性
    parser.add_argument('--Mean_evaluator', action='store_true', default=False) # 用Mean训练的评估器来train.py里测GT指标
    parser.add_argument('--using_meta', action='store_true', default=False) # train.py里使用meta均值来训练，发现在MDM上不行因为放大倍数太大，在VQ的方法里是可以的
    parser.add_argument('--l_smooth', type=float, default=0)
    parser.add_argument('--test_gt_metric', action='store_true', default=False)
    parser.add_argument('--adapt_el_tcl', action='store_true', default=False)
    parser.add_argument('--gen_with_text_grad', action='store_true', default=False)
    parser.add_argument('--unlock_motion_enc', action='store_true', default=False)
    parser.add_argument('--bone_loss', type=float, default=0)
    parser.add_argument('--ric_rot_loss', type=float, default=0)
    parser.add_argument('--salad_vaeenc_loss', type=float, default=0)
    parser.add_argument('--init_noisy_data_level', type=int, default=0)
    parser.add_argument('--train_test_set', action='store_true', default=False)
    parser.add_argument('--eval_sample_num', type=int, default=10000)
    parser.add_argument('--test_with_noisy_step', type=int, default=0)
    parser.add_argument('--evaluator_train', type=str, default=None) # 训练用的评估器
    parser.add_argument('--evaluator_eval', type=str, default=None) # 验证用的评估器
    parser.add_argument('--train_max_noisy_step', type=int, nargs='+', default=[0])
    # parser.add_argument('--train_max_noisy_step', type=int, default=0)
    parser.add_argument('--init_eval', action='store_true', default=False)
    parser.add_argument('--eval_during_train', action='store_true', default=False)
    parser.add_argument('--vis_during_train', action='store_true', default=False, help='在训练验证时可视化 SnapMoGen motion 并保存为 MP4（仅 dataset_name=snapmogen 时生效）')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--only_emb_loss', action='store_true', default=False)
    parser.add_argument('--text_cos_loss', type=float, default=0)
    parser.add_argument('--text_emb_loss', type=float, default=0)
    parser.add_argument('--text_infonce_loss', type=float, default=0)
    parser.add_argument('--infonce_loss', type=float, default=0)
    parser.add_argument('--ric_global_loss', type=float, default=0)
    
    parser.add_argument('--dist_loss', type=float, default=0)
    parser.add_argument('--cos_loss', type=float, default=0)
    parser.add_argument('--emb_loss', type=float, default=0)
    parser.add_argument('--use_cache', action='store_true', default=False)
    parser.add_argument('--datatype', type=str, choices=['hml', 'smpl'], default='smpl')
    parser.add_argument('--num_noisy_timesteps', type=int, default=10)
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--timestep_respacing', type=str, default='100')
    parser.add_argument('--use_ddim', type=int, choices=[1, 0], default=0)
    parser.add_argument('--only_t2m_s2', type=int, default=0)
    parser.add_argument('--bfgs_type', type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--guidance_param', type=float, default=2.5)
    parser.add_argument('--bfgs_lr', type=float, default=0.5) 
    parser.add_argument('--gtric_fortest', action='store_true', default=False)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--root_zero_grad', type=int, default=0)
    parser.add_argument('--replication_times', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train_posterior', action='store_true', default=False)
    parser.add_argument('--use_lbfgs', type=int, choices=[1, 0], default=0)
    parser.add_argument('--stage2_no_root_y', type=int, choices=[0, 1], default=0)
    parser.add_argument('--return_type', type=str, choices=['x0', 'sample', 'priorMDM'], default='sample')
    parser.add_argument("--max_samples", default=10000, type=int)
    parser.add_argument("--cond_mode", default='both_text_spatial', type=str)
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l2')
    # parser.add_argument("--control_joint", default=-1, type=int, help='-1 means randomly choose a joint')
    parser.add_argument('--control_joint', default=[-1], nargs="+", type=int)
    parser.add_argument('--density',  type=int, default=0)
    parser.add_argument('--attn_mask', action='store_true', default=False)

    parser.add_argument('--root_dist_loss', action='store_true', default=False, help='instead of element-wise loss, use global loss for root')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'debug'], default='train')
    parser.add_argument('--multi_joint_control', action='store_true', default=False)
    parser.add_argument('--temporal_complete', type=float, default=0.0, help='whether add temporal completion')
    parser.add_argument('--normalize_traj', action='store_true', default=True)
    parser.add_argument('--note', type=str, default='this is note')
    parser.add_argument('--num_layers_E', type=int, default=3)
    parser.add_argument('--num_layers_D', type=int, default=1)
    parser.add_argument('--roottype', type=str, default=None) 
    parser.add_argument('--modeltype', type=str, default=None) 
    parser.add_argument('--gpu', nargs='+', default=['0'])
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit', 'snapmogen'], help='dataset directory')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    
    ## optimization
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[300000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    # training settings
    parser.add_argument("--resume_root", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume_trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument('--out_dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out_dir')
    parser.add_argument('--print_iter', default=100, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--save_iter', default=5000, type=int, help='save frequency')
    parser.add_argument('--total_iter', default=600000, type=int, help='number of total iterations to run')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--latent_dim", type=int, default=512, help="latent dimension in transformer")
    

    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')

    return parser.parse_args()