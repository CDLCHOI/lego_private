import torch
import options.option_transformer as option_trans
import os 
import numpy as np
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
# fixseed(123456) 
# fixseed(33333) 

from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt, vis_motion
from utils.model_util import create_gaussian_diffusion_simple, sample_ADControl, sample_omni263mdm_fuse, sample_omnicontrol
from utils.lora_util import load_lora_mdm_for_eval
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from utils.visualize.vis_utils import npy2obj


def plot_t2m(data, save_dir, captions, m_lengths, save_npy=False, save_npz=False, start_idx=0):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = gen_loader.dataset.t2m_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, motion_data) in enumerate(zip(captions, data)):
        motion_data = motion_data[:m_lengths[i]]
        joints = recover_from_ric(torch.from_numpy(motion_data).float(), 22).numpy()

        save_mp4_path = os.path.join(save_dir, f'{start_idx+i:03d}.mp4')
        plot_3d_motion(save_mp4_path, t2m_kinematic_chain, joints, title=caption, fps=20)
        print(f'finish saving {save_mp4_path}')

        if save_npy:
            save_npy_path = os.path.join(save_dir, f'{start_idx+i:03d}.npy')
            save_dict = {'motions': joints, 'real_length': m_lengths[i]}
            np.save(save_npy_path, save_dict)

        if save_npz:
            joints = joints[None,:].transpose(0, 2, 3, 1) # (b,22,3,196)
            data = {}
            data['real_length'] = m_lengths[i].item()
            data['motions'] = joints[..., :m_lengths[i].item()]
            data['text'] = clip_text[i]
            
            npy_to_obj = npy2obj('',device=0, cuda=True, data=data)
            save_npz_path = os.path.join(save_dir, f'{start_idx+i:03d}.npz')
            npy_to_obj.save_npz(save_npz_path)
            print(f' save {save_npz_path}')


def build_model(args, modeltype, ckpt_path):
    if modeltype in ['mdm']:
        from models.mdm.model import MDM
        from utils.model_util import get_mdm_args
        net = MDM(**get_mdm_args(args, modeltype))
    elif modeltype in ['mdm_bert']:
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, modeltype))
    else:
        raise NotImplementedError(f"modeltype {modeltype} not implemented")

    load_lora_mdm_for_eval(net, ckpt_path, args.add_clip_lora)
    diffusion = create_gaussian_diffusion_simple(args, net, modeltype)
    net.cuda()
    net.eval()
    return net, diffusion

if __name__ == '__main__':
    args.dataset_name = 't2m'
    args.batch_size = 1
    args.diffusion_steps = 50

    # 原MDM
    # args.text_encoder_type = 'clip'; args.modeltype = 'mdm_bert'; args.add_clip_lora = False
    # args.resume_trans = 'output/0814_MDMCLIP_b128/net_best.pth'
    # args.no_random = True

    # ours
    args.text_encoder_type = 'clip'; args.modeltype = 'mdm_bert'; args.add_clip_lora = True
    # args.resume_trans = 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/net_best.pth'
    args.resume_trans = 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ricglobal1/net_best.pth'
    args.no_random = True


    base_dir = os.path.dirname(args.resume_trans)
    vis_dir = os.path.join(base_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    if args.modeltype =='mdm_bert':
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))
    else:
        raise NotImplementedError

    if 'lora' in args.resume_trans:
        load_lora_mdm_for_eval(net, args.resume_trans)
    else:
        load_ckpt(net, args.resume_trans, key=None, strict=False)

    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    net.cuda()
    net.eval()

    # net, diffusion = build_model(args, args.modeltype, args.resume_trans)

    

    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)

        
    for i, batch in enumerate(gen_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        # real_length = torch.Tensor([196]).to(gt_motion.device).repeat(b).int()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        # real_length = torch.Tensor([196]).to(gt_motion.device)

        clip_text = ('A person is standing. At first, his hands are hanging down. Then, his right hand rises from below and upwards, making a greeting gesture.',)
        model_kwargs = {}
        model_kwargs['clip_text'] = clip_text
        model_kwargs['real_mask'] = real_mask
        # print('clip_text = ', clip_text)
        
        x0 = gt_motion


        # 生成
        # with open(os.path.join(vis_dir, f'{i}.txt'),'a') as f:
        #     f.write(str(clip_text))
        sample = diffusion.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=args.batch_size)
        
        # 蓝色的生成的，红色是GT
        for j in range(args.batch_size):
            save_path = os.path.join(vis_dir, f'{j}.html')
            vis_motion(motion1=sample[j], motion2=None, save_path=save_path, vis=True)
            # vis_motion(motion1=sample[j], motion2=x0[j], save_path=save_path, vis=True)
        print('filename = ', filename)
        # plot_t2m(sample, vis_dir, [f+' '+c+' '+str(l.item()) for c,f,l in zip(clip_text,filename,real_length)], real_length, save_npz=True, start_idx=i*args.batch_size)

        if i==20:
            break

        

    

    
    
    
    