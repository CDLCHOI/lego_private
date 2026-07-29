import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)  # 设定GPU
os.environ['OMP_NUM_THREADS'] = '8'

import torch
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.enabled = False
import torch.nn as nn
import numpy as np
import ipdb

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip
import json
from utils.model_util import initial_optim, get_logger, LearnableLossWeights
from utils.mask_utils import load_ckpt, load_lora_ckpt
from dataset import dataset_control, dataset_critic
import warnings
warnings.filterwarnings('ignore')
import shutil
from utils.fixseed import fixseed
from glob import glob

if __name__ == '__main__':
    cmd = "python " + " ".join(sys.argv)
    # if args.modeltype == 'mdm_bert':
    #     args.diffusion_steps = 50

    if args.pretrained_lora_path:
        args.add_clip_lora = True

    fixseed(args.seed)

    # SnapMoGen 数据集特殊设置
    if args.dataset_name == 'snapmogen':
        if args.max_motion_length == 196:  # 未通过命令行显式指定
            args.max_motion_length = 320
            print(f'[SnapMoGen] auto-set max_motion_length = {args.max_motion_length}')

    # 训练前准备
    args.out_dir = pjoin(args.out_dir, args.exp_name) # output/trans_exp_name
    if args.overwrite and os.path.exists(args.out_dir):
        # assert not os.path.exists(pjoin(args.out_dir, 'net_last.pth')), f'net_last.pth exist in {args.out_dir}'
        assert glob(args.out_dir + '/*.pth') == [], f'already exist checkpoints in {args.out_dir}'
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(cmd)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True)) # args所有输出到log
    logger.info(args.note)
    torch.manual_seed(args.seed)

    # # mean and std
    # humanml_mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
    # humanml_std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
    
    # # CLIP
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    # # clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # clip_model.eval()
    # for p in clip_model.parameters():
    #     p.requires_grad = False

    # class TextCLIP(torch.nn.Module):
    #     def __init__(self, model) :
    #         super(TextCLIP, self).__init__()
    #         self.model = model
            
    #     def forward(self,text):
    #         with torch.no_grad():
    #             word_emb = self.model.token_embedding(text).type(self.model.dtype)
    #             word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
    #             word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
    #             word_emb = self.model.transformer(word_emb)
    #             word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
    #             enctxt = self.model.encode_text(text).float()
    #         return enctxt, word_emb
    # clip_model = TextCLIP(clip_model)

    if args.modeltype in ['mdm']:
        from models.mdm.model import MDM
        from utils.model_util import get_mdm_args
        net = MDM(**get_mdm_args(args, args.modeltype))
    elif args.modeltype in ['mdm_bert']:
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))
    elif args.modeltype in ['salad']:
        from models.salad.salad import SALAD
        net = SALAD()
        # load_ckpt(net, '/home/deli/project/salad/checkpoints/t2m/t2m_vae_gelu/model/net_best_fid.tar', key='vae', strict=False)
        # load_ckpt(net, '/home/deli/project/salad/checkpoints/t2m/t2m_denoiser_vpred_vaegelu/model/net_best_fid.tar', key='denoiser', strict=False)
    elif args.modeltype in ['critic']:
        from models.critic.critic import MotionCritic
        net = MotionCritic(depth=1, dim_feat=256, dim_rep=512, mlp_ratio=4, num_joints=22+1 if args.dataset_name == 't2m' else 21+1)
    elif args.modeltype in ['mdmcritic']:
        from models.mdm_critic import MDMCritic
        from utils.model_util import get_mdmcritic_args
        net = MDMCritic(**get_mdmcritic_args(args))
    else:   
        raise ValueError("modeltype not found")

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)

    try:
        logger.info(f'load_ckpt(): key=None')
        load_ckpt(net, args.resume_trans, key=None, strict=False, filter=True)
    except:
        logger.info(f'load_ckpt(): key=model')
        load_ckpt(net, args.resume_trans, key='base', strict=False, filter=True) # 读郭岭的MDMdec用
    # except:
    #     logger.info(f'load_lora_ckpt():')
    #     load_lora_ckpt(net, args.resume_trans)
            
    if sys.gettrace():
        net.eval(); logger.info(' net is eval !!!!!!!')
        # net.train(); logger.info(' net is train ~~~~~')
    else:
        net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()

    if args.no_random:
        shuffle=False; print('______shuffle False ____  debug debug debug ===')
    else:
        shuffle=True 

    train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode, diffusion=diffusion, shuffle=shuffle)
    train_loader_iter = dataset_control.cycle(train_loader)

    val_batch = 100 if args.dataset_name == 'snapmogen' else 32
    logger.info(f'val_batch = {val_batch} for dataset {args.dataset_name}')
    gt_loader = dataset_control.DataLoader(batch_size=val_batch, args=args, mode='gt', split='test', shuffle=shuffle, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=val_batch, args=args, mode='eval', split='test', shuffle=shuffle, num_workers=0, drop_last=True) # 这里shuffle=False是为了保证每一次训练中验证都是同批数据
    logger.info(f'gen_loader shuffle = {shuffle}')
    
    if args.ablation_separate_update:
        optimizer = torch.optim.AdamW([param for name, param in net.named_parameters() if not 'clip' in name], lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizer_clip = torch.optim.AdamW(net.module.clip_model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    
    if args.modeltype in ['mdm', 'salad', 'mdm_bert']:
        diffusion.log_file = os.path.join(args.out_dir, 'run.log')
        diffusion.gt_loader = gt_loader
        diffusion.gen_loader = gen_loader
        diffusion.writer = writer
        diffusion.adapt_weights = adapt_weights  if args.adapt_el_tcl else 0
        diffusion.optimizer_adapt = optimizer_adapt if args.adapt_el_tcl else 0
        diffusion.trainer_func_mdm(train_loader_iter, logger, optimizer, scheduler)
    elif args.modeltype in ['critic', 'mdmcritic']:
        from models.critic import critic_trainer
        critic_trainer.trainer_func(net, train_loader_iter, logger, optimizer, scheduler, args)

    
    

