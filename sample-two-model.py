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
from utils.model_util import create_gaussian_diffusion_simple, get_clip_model, sample_ADControl, sample_omni263mdm_fuse, sample_omnicontrol
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

    load_lora_mdm_for_eval(net, args.resume_trans, args.add_clip_lora)
    diffusion = create_gaussian_diffusion_simple(args, net, modeltype)
    net.cuda()
    net.eval()

    return net, diffusion

if __name__ == '__main__':
    args.dataset_name = 't2m'
    args.batch_size = 1
    dir_path = 'visualization/'
    args.resume_trans3 = None

    # args.use_ddim = 1; 
    args.diffusion_steps = 1000
    # 对照组，50步，干净数据训练，有无el
    # args.resume_trans = 'output/humanml_enc_512_50steps/model000750000.pt'; args.diffusion_steps = 50
    # args.resume_trans2 = 'output/0605_mdm_el50_50step_find_bestfid/net_best.pth'; args.diffusion_steps = 50
    # args.resume_trans3 = None
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_clean_w_wo_el/'


    # 对照组  0-50随机加噪
    # args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'
    # args.resume_trans1 = 'output/0528_mdm_step50_noisy50/net_last.pth'; args.diffusion_steps = 50
    # args.resume_trans2 = 'output/0528_mdm_el50_step50_noisy50/net_last.pth'; args.diffusion_steps = 50
    # args.resume_trans3 = None
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_random0_50_noisy_w_wo_el/'

    # 对照组 固定10步加噪训练
    # args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'; args.diffusion_steps = 50
    # args.resume_trans1 = 'output/0531_mdm_step50_noisy1010/net_last.pth';
    # args.resume_trans1 = 'output/0531_mdm_step50_noisy1010_scratch/net_best.pth' # 应该用这个，这个是从头训的
    # args.resume_trans2 = 'output/0531_mdm_step50_noisy1010_el50/net_best.pth'
    # args.resume_trans3 = 'output/0601_mdm_step50_noisy1010_el50_union/net_last.pth'
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_w_wo_el/'

    # mdm_bert模型，有无el的对比，查看人体变形情况
    # args.modeltype1 = 'mdm_bert'; args.modeltype2 = 'mdm_bert'; args.diffusion_steps = 50
    # args.resume_trans1 = 'output/0609_MDMBERT/net_best.pth'
    # args.resume_trans2 = 'output/0618_MDMBERT_el10/net_best.pth'

    # mdm_bert模型，el 与 el+tcl 对比
    args.modeltype1 = 'mdm_bert'; args.modeltype2 = 'mdm_bert'; args.diffusion_steps = 50
    args.resume_trans1 = 'output/0618_MDMBERT_el10/net_best.pth'
    args.resume_trans2 = 'output/0708_MDMBERT_el10_tcl1_infoeval/net_best.pth'

    # 仅对数据集做一次加噪的实验
    # args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'; args.diffusion_steps = 50
    # args.resume_trans1 = 'output/0619_mdm_step50_noisyonce5/net_best.pth'
    # args.resume_trans2 = 'output/0619_mdm_step50_noisyonce5_el50/net_best.pth'
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_noisyonce5_w_wo_el/'

    # mdm_bert模型，仅tcl与tcl+el对比，原评估器
    # args.modeltype1 = 'mdm_bert'; args.modeltype2 = 'mdm_bert'; args.diffusion_steps = 50
    # args.resume_trans1 = 'output/0707_MDMBERT_tcl1_orieval/net_best.pth'
    # args.resume_trans2 = 'output/0707_MDMBERT_el10_tcl1_orieval/net_best.pth' 
    # dir_path = f'visualization/compare_{args.diffusion_steps}tcl_and_eltcl/'

    # mdm_bert模型，仅tcl与tcl+el对比，InfoNCE评估器
    args.modeltype1 = 'mdm_bert'; args.modeltype2 = 'mdm_bert'; args.diffusion_steps = 50
    # args.resume_trans1 = 'output/0707_MDMBERT_tcl1_orieval/net_best.pth'
    args.resume_trans1 =  'output/0711_MDMBERT_el10_tcl1_infoeval/net_best.pth' # 加了平滑loss，但是FID略差的
    args.resume_trans2 = 'output/0708_MDMBERT_el10_tcl1_infoeval/net_best.pth' 
    # dir_path = f'visualization/compare_{args.diffusion_steps}tcl_and_eltcl/'


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    
    net1, diffusion1 = build_model(args, args.modeltype1, args.resume_trans1)
    net2, diffusion2 = build_model(args, args.modeltype2, args.resume_trans2)

    # from utils.model_util import get_mdm_args1
    # from models.mdm import MDM
        
    # net1 = MDM(**get_mdm_args(args))
    # load_ckpt(net1, args.resume_trans, key=None, strict=True)
    # diffusion = create_gaussian_diffusion_simple(args, net1, args.modeltype)
    # net1.cuda()
    # net1.eval()

    # net2 = MDM(**get_mdm_args(args))
    # load_ckpt(net2, args.resume_trans2, key=None, strict=True)
    # diffusion2 = create_gaussian_diffusion_simple(args, net2, args.modeltype)
    # net2.cuda()
    # net2.eval()

    if args.resume_trans3 is not None:
        net3, diffusion3 = build_model(args, args.modeltype3, args.resume_trans3)
    

    gen_loader = dataset_control.DataLoader(batch_size=1, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    
    for i, batch in enumerate(gen_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)

        model_kwargs = {}
        model_kwargs['clip_text'] = clip_text
        model_kwargs['real_mask'] = real_mask
        print('clip_text = ', clip_text)
        print('real_length = ', real_length)
        
        x0 = gt_motion

        # 将文本存到txt里
        # with open(os.path.join(dir_path, f'{i}.txt'),'a') as f:
        #     f.write(str(clip_text))

        # 生成
        # t, weights = diffusion1.schedule_sampler.sample1(b, gt_motion.device) # timestep
        sample1 = diffusion1.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        sample2 = diffusion2.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        if args.resume_trans3 is not None:
            sample3 = diffusion3.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=1)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        text = clip_text[0].replace('.','').replace(' ','_')
        # save1 = os.path.join(dir_path, f'{i}.html')
        # save2 = os.path.join(dir_path, f'{i}_el.html'); assert 'el' in args.resume_trans2
        save_12 = os.path.join(dir_path, f'net_both_{i}.html')
        # save_13 = os.path.join(dir_path, f'{i}_13.html')
        # motion1蓝色 ， motion2 红色
        # vis_motion(pred_motion=sample1[0], gt_motion=x0[0], dir_path=save1, vis=False)
        # vis_motion(pred_motion=sample2[0], gt_motion=x0[0], dir_path=save2, vis=False)
        vis_motion(motion1=sample2[0], motion2=sample1[0], save_path=save_12, vis=True)

        # 查看2个模型的ric和rot匹配度
        # save1 = os.path.join(dir_path, f'net1_{i}.html')
        # vis_motion(motion1=sample1[0], motion2=None, save_path=save1, vis=False, using_rotation=False, joint2_from_joint1rot=True)
        # save2 = os.path.join(dir_path, f'net2_{i}.html')
        # vis_motion(motion1=sample2[0], motion2=None, save_path=save2, vis=False, using_rotation=False, joint2_from_joint1rot=True)

        
        if i==5:
            break


        

    

    
    
    
    