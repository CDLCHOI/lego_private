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
    vis_dir = 'visualization'
    # vis_dir = os.path.join(base_dir, 'visualization')
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

    mean = np.load('./dataset/HumanML3D/Mean.npy')
    std = np.load('./dataset/HumanML3D/Std.npy')

    clip_text = (
        'A man runs forward',
        'A person runs in a straight line',
        'The person runs forward quickly',
        # 'A person faces forward and walks backward in a straight line',
        # 'A person faces forward and walks backward',
        # 'A person faces forward and walks forward in a straight line',
        # 'A person faces forward and walks forward',
        # 'A man waves right hand.',
        # 'A person is waving his right hand.',
        # 'A person is standing. At first, his hands are hanging down. Then, his right hand rises from below and upwards, making a greeting gesture.',
        # 'A person stands with his hands hanging down. Then his right hand rises from below and upwards, making a greeting gesture.',
        # 'A person standing and waving with the right hand.',
        # 'Someone is waving their right hand to say hello.',
        # 'A man performs a right-hand waving gesture while standing.',
        # 'A person stands still and waves their right hand.',
        # 'Standing upright, a person lifts their right arm and swings the hand side to side.',
        # 'A person starts in a neutral standing pose, then elevates their right hand to shoulder height to wave.',
        # 'A man stands in place, waving his right hand as if greeting a friend',
        )
    args.batch_size = len(clip_text)
    real_length = torch.Tensor([196,]).int().cuda()
    real_mask = generate_src_mask(196, real_length) # (b,196)
    real_mask = real_mask.repeat(args.batch_size, 1)

    model_kwargs = {}
    model_kwargs['clip_text'] = clip_text
    model_kwargs['real_mask'] = real_mask
    # print('clip_text = ', clip_text)
    


    # 生成
    # with open(os.path.join(vis_dir, f'{i}.txt'),'a') as f:
    #     f.write(str(clip_text))
    sample = diffusion.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=args.batch_size)
    sample = sample.cpu().numpy() * std + mean
    joints = recover_from_ric(torch.from_numpy(sample), 22).numpy()


    # 蓝色的生成的，红色是GT
    for j in range(args.batch_size):
        save_path = os.path.join(vis_dir, f'{j}.html')
        save_mp4_path = os.path.join(vis_dir, f'{j}.mp4')
        # vis_motion(motion1=sample[j], motion2=None, save_path=save_path, vis=False)
        plot_3d_motion(save_mp4_path, t2m_kinematic_chain, joints[j], title=clip_text[j], fps=20)
        print(f'save {save_mp4_path}')
        # plot_t2m(sample, vis_dir, clip_text, real_length, save_npz=True, start_idx=j)

        

    

    
    
    
    