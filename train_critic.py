import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)  # 设定GPU
os.environ['OMP_NUM_THREADS'] = '8'

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import numpy as np
import ipdb

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip
import json
from utils.model_util import initial_optim, get_logger
from utils.mask_utils import load_ckpt
from dataset import dataset_control, dataset_critic
import warnings
warnings.filterwarnings('ignore')
import shutil
from utils.fixseed import fixseed

if __name__ == '__main__':
    # fixseed(args.seed)
    # 训练前准备
    args.out_dir = pjoin(args.out_dir, args.exp_name) # output/trans_exp_name
    if args.overwrite and os.path.exists(args.out_dir):
        assert not os.path.exists(pjoin(args.out_dir, 'net_last.pth')), f'net_last.pth exist in {args.out_dir}'
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True)) # args所有输出到log
    logger.info(args.note)
    torch.manual_seed(args.seed)


    if args.modeltype in ['critic']:
        from models.critic.critic import MotionCritic
        net = MotionCritic(depth=1, dim_feat=256, dim_rep=512, mlp_ratio=4, num_joints=22+1 if args.dataset_name == 't2m' else 21+1)
    elif args.modeltype in ['mdmcritic']:
        from models.mdm_critic import MDMCritic
        from utils.model_util import get_mdmcritic_args
        net = MDMCritic(**get_mdmcritic_args(args))
    else:   
        raise ValueError("modeltype not found")

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, None)


    load_ckpt(net, args.resume_trans, key=None, strict=True)
            
    if sys.gettrace():
        net.eval(); logger.info(' net is eval !!!!!!!')
    else:
        net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()

    train_loader = dataset_critic.DataLoader(args, diffusion, split='train')
    train_loader_iter = dataset_critic.cycle(train_loader)

    # 训练配置
    optimizer = initial_optim(args.lr, args.weight_decay, net, args.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    
    from models.critic import critic_trainer
    critic_trainer.trainer_func(net, train_loader_iter, logger, optimizer, scheduler, args)

    
    

