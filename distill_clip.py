import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)  # 设定GPU
os.environ['OMP_NUM_THREADS'] = '8'

import torch
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
    if args.modeltype == 'mdm_bert':
        args.diffusion_steps = 50

    if args.pretrained_lora_path:
        args.add_clip_lora = True

    # fixseed(args.seed)

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


    

    if args.modeltype in ['mdm_bert']:
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)

    try:
        logger.info(f'load_ckpt(): key=None')
        load_ckpt(net, args.resume_trans, key=None, strict=False, filter=True)
    except:
        logger.info(f'load_ckpt(): key=model')
        load_ckpt(net, args.resume_trans, key='model', strict=False, filter=True) # 读郭岭的MDMdec用
    # except:
    #     logger.info(f'load_lora_ckpt():')
    #     load_lora_ckpt(net, args.resume_trans)
            
    net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()

    train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode, diffusion=diffusion)
    train_loader_iter = dataset_control.cycle(train_loader)

    if args.no_random:
        shuffle=False; print('______shuffle False ____  debug debug debug ===')
    else:
        shuffle=True 
    
    gt_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='gt', split='test', shuffle=shuffle, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='eval', split='test', shuffle=shuffle, num_workers=0, drop_last=True) # 这里shuffle=False是为了保证每一次训练中验证都是同批数据
    logger.info(f'gen_loader shuffle = {shuffle}')


    # 训练配置
    # optimizer = initial_optim(args.lr, args.weight_decay, net, args.optimizer)
    if args.adapt_el_tcl:
        assert args.emb_loss == 0 and args.text_cos_loss == 0 and args.text_cos_loss == 0, f'el {args.emb_loss}, tcl {args.text_cos_loss}, tel {args.text_emb_loss}'
        adapt_weights = LearnableLossWeights(np.log(1), np.log(1))
        optimizer_adapt = torch.optim.AdamW(adapt_weights.parameters(), lr=1e-2, betas=(0.5, 0.9), weight_decay=args.weight_decay)
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

    
    

