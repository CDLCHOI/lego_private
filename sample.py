import torch
import options.option_transformer as option_trans
import os 
import numpy as np
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
# fixseed(123456) # 站着挥右手
# fixseed(111) # 直走路
# fixseed(3722241)
from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt, vis_motion
from utils.model_util import create_gaussian_diffusion_simple, sample_ADControl, sample_omni263mdm_fuse, sample_omnicontrol

if __name__ == '__main__':
    args.dataset_name = 't2m'
    args.batch_size = 1

    args.diffusion_steps = 50

    # args.modeltype = 'mdm'; args.resume_trans = 'output/0531_mdm_step50_noisy1010_el50/net_last.pth'
    # args.modeltype = 'mdm'; args.resume_trans = 'output/0619_mdm_step50_noisyonce5/net_best.pth'
    # args.modeltype = 'mdm'; args.resume_trans = 'output/0619_mdm_step50_noisyonce5_el50/net_last.pth'
    # args.modeltype = 'mdm'; args.resume_trans = 'output/0531_mdm_step50_noisy1010_scratch/net_best.pth'
    
    # args.resume_trans = 'output/0619_mdm_step50_noisyonce5_el50/net_last.pth'; 
    # args.modeltype = 'mdm_bert'; args.resume_trans = 'output/0609_MDMBERT_el50_pretrain2/net_best.pth'
    # args.modeltype = 'mdm_bert'; args.resume_trans = 'output/0612_MDMBERT_el50_tel50/net_best.pth'
    # args.modeltype = 'mdm_bert'; args.resume_trans = 'output/0807_MDMBERT_cl1_tcl1_0807/net_best.pth'


    base_dir = os.path.dirname(args.resume_trans)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    if args.modeltype == 'mdm':
        from utils.model_util import get_mdm_args
        from models.mdm.model import MDM
        net = MDM(**get_mdm_args(args))
    elif args.modeltype =='mdm_bert':
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))
    else:
        raise NotImplementedError

        
    load_ckpt(net, args.resume_trans, key=None, strict=False)
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    net.cuda()
    net.eval()

    

    gen_loader = dataset_control.DataLoader(batch_size=1, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    dir_path = f'visualization/noisy_by_{args.diffusion_steps}step/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    for i, batch in enumerate(gen_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        real_length = torch.Tensor([196]).to(gt_motion.device)

        model_kwargs = {}
        model_kwargs['clip_text'] = clip_text
        model_kwargs['real_mask'] = real_mask
        print('clip_text = ', clip_text)
        
        x0 = gt_motion



        # # 存加噪样本的
        # with open(os.path.join(dir_path, f'{i}.txt'),'a') as f:
        #     f.write(str(clip_text))
        # for a in [2,4,6,8,10]:
        #     t = torch.tensor(np.random.randint(a, a+1)).cuda()
        #     xt = diffusion.q_sample(x0, t) 
        #     save_path = os.path.join(dir_path, f'{i}_max{args.diffusion_steps}_noisy{t.item()}.html')
        #     # 蓝色的加噪的，红色是GT
        #     vis_motion(pred_motion=xt[0], gt_motion=x0[0], save_path=save_path, vis=False)
        # if i==6:
        #     break

        # 生成
        with open(os.path.join(base_dir, f'{i}.txt'),'a') as f:
            f.write(str(clip_text))
        t, weights = diffusion.schedule_sampler.sample(b, gt_motion.device) # timestep
        sample = diffusion.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        save_path = os.path.join(base_dir, '1.html')
        # 蓝色的生成的，红色是GT
        vis_motion(motion1=sample[0], motion2=x0[0], save_path=save_path, vis=True)
        if i==5:
            break

        

    

    
    
    
    