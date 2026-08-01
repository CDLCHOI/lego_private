import torch
import options.option_transformer as option_trans
import os 
import numpy as np
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
# fixseed(123456) # 站着挥右手
# fixseed(111) # 直走路
# fixseed(3722241)
from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt, vis_motion
from utils.model_util import create_gaussian_diffusion_simple
from utils.lora_util import load_lora_mdm_for_eval

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
    args.no_random = True
    args.dataset_name = 't2m'
    args.batch_size = 1
    args.diffusion_steps = 50
    args.text_encoder_type = 'clip';args.modeltype = 'mdm_bert'; 
    args.resume_trans1 = 'output/0814_MDMCLIP_b128/net_best.pth'
    
    net1, diffusion1 = build_model(args, args.modeltype, args.resume_trans1)

    args.add_clip_lora = True
    args.resume_trans2 = 'output/0911_MDMCLIP_preatrainlora_ric1_b64/net_best.pth'
    net2, diffusion2 = build_model(args, args.modeltype, args.resume_trans2)
    

    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    dir_path = f'visualization/0814_two/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    for i, batch in enumerate(gen_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        real_length = torch.Tensor([196]).to(gt_motion.device)
        real_mask = generate_src_mask(max_length, real_length) # (b,196)

        model_kwargs = {}
        model_kwargs['clip_text'] = clip_text
        model_kwargs['real_mask'] = real_mask
        print('clip_text = ', clip_text)
        
        x0 = gt_motion

        # 生成
        with open(os.path.join(dir_path, f'{i}.txt'),'a') as f:
            f.write(str(clip_text))
            
        sample1 = diffusion1.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        sample2 = diffusion2.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        
        save_12 = os.path.join(dir_path, f'net_both_{i}.html')

        # motion1蓝色 ， motion2 红色
        vis_motion(motion1=sample2[0], motion2=sample1[0], save_path=save_12, vis=True)
        
        if i==5:
            break

        

    

    
    
    
    