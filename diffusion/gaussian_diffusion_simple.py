import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as pjoin
from diffusion.respace import space_timesteps
from utils.motion_process import recover_from_ric, recover_from_rotation, t2m_tgt_skel, t2m_tgt_offsets, kit_tgt_skel, kit_tgt_offsets
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper, EvaluatorMARDM
from data_loaders.humanml.networks.evlauator_wrapper_salad import EvaluatorModelWrapperSALAD
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
from utils.mask_utils import generate_src_mask, vis_motion, recover_from_rotation
from utils.metrics import mean_joint_error_torch
from models.cfg_sampler import ClassifierFreeSampleModel, CFG_SALAD
import ipdb
from models.Contrastive import ContrastiveLoss
from models.modules import InfoNCELoss
import loratorch as lora

from utils.fixseed import fixseed


class GaussianDiffusionSimple:
    def __init__(self, args, model, modeltype, clip_model, betas) -> None:
        self.args = args
        self.modeltype = modeltype # 'ED'
        self.model = model
        if self.modeltype in ['salad']:
            self.model_cfg = CFG_SALAD(self.model, 7.5)
        else:
            self.model_cfg = ClassifierFreeSampleModel(self.model, self.args.guidance_param)
        self.clip_model = clip_model
        self.infonce_loss = InfoNCELoss()
        self.adapt_weights = 0
        self.optimizer_adapt = 0

        # 训练用的评估器
        if self.args.dataset_name == 'snapmogen':
            if args.snapmogen_evaluator_train_type == 'trans':
                # 使用 SnapMoGen 的 evaluator（本地模块）
                from models.snapmogen_evaluator import EvaluatorWrapper
                from utils.config_utils import load_config
                # 加载 SnapMoGen evaluator 配置（evaluator的dim_pose需要与checkpoint一致，此处为148）
                eval_cfg = load_config('./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml')
                eval_model_path = self.args.evaluator_train
                if eval_model_path is None:
                    eval_model_path = './SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/model/net_best_top1.tar'
                self.eval_wrapper = EvaluatorWrapper(eval_cfg, device=torch.device('cuda'), model_path=eval_model_path)
                self.eval_wrapper.eval()
            elif args.snapmogen_evaluator_train_type == 'gru':
                if self.args.evaluator_train is not None:
                    print('=== using GRU evaluator train for SnapMoGen ===')
                    self.eval_wrapper = EvaluatorMDMWrapper(self.args.dataset_name, torch.device('cuda'), self.args, self.args.evaluator_train)
                    self.eval_wrapper.eval()
                else:
                    print('=== evaluator will not be used during training ===')
            
            
        else:
            if self.args.evaluator_train_type =='gru':
                print('=== args.evaluator_train is None, using default evaluator')
                self.eval_wrapper = EvaluatorMDMWrapper(self.args.dataset_name, torch.device('cuda'), self.args, self.args.evaluator_train)
            elif self.args.evaluator_train_type =='tmr':
                pass

        # if self.args.unlock_motion_enc:
        #     self.optimizer_movement_enc = torch.optim.AdamW(self.eval_wrapper.movement_encoder.parameters(),
        #                                                     lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        #     self.optimizer_motion_enc = torch.optim.AdamW(self.eval_wrapper.motion_encoder.parameters(),
        #                                                     lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        #     self.optimizer_list_enc = [self.optimizer_movement_enc, self.optimizer_motion_enc]

        #     self.scheduler_movement_enc = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_movement_enc, milestones=args.lr_scheduler, gamma=args.gamma)
        #     self.scheduler_motion_enc = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_motion_enc, milestones=args.lr_scheduler, gamma=args.gamma)
        #     self.scheduler_list_enc = [self.scheduler_movement_enc, self.scheduler_motion_enc]

        

        self.contrastive_loss = ContrastiveLoss(margin=3.0)
                
        self.gt_loader = None
        self.gen_loader = None
        self.log_file = None
        self.writer = None

        if self.args.dataset_name == 't2m':
            self.n_joints = 22
            self.mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
            self.raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        elif self.args.dataset_name == 'kit':
            self.n_joints = 21
            self.mean = torch.from_numpy(np.load('dataset/KIT-ML/Mean.npy')).cuda()[None, None, ...].float() # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/KIT-ML/Std.npy')).cuda()[None, None, ...].float()
            self.raw_mean = torch.from_numpy(np.load('dataset/kit_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,21,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/kit_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,21,3)
        elif self.args.dataset_name == 'snapmogen':
            self.n_joints = 24
            # SnapMoGen 使用其数据集目录下的 mean.npy 和 std.npy
            snapmogen_mean = np.load('/data/motion/SnapMoGen/meta_data/mean.npy')
            snapmogen_std = np.load('/data/motion/SnapMoGen/meta_data/std.npy')
            self.mean = torch.from_numpy(snapmogen_mean).cuda()[None, None, ...].float()
            self.std = torch.from_numpy(snapmogen_std).cuda()[None, None, ...].float()
            # SnapMoGen 的 raw_mean 和 raw_std 暂时使用默认值
            self.raw_mean = torch.zeros(1, 1, 24, 3).cuda()
            self.raw_std = torch.ones(1, 1, 24, 3).cuda()

        

        # diffusion相关参数值
        betas = np.array(betas, dtype=np.float64) # 每个step的噪声方差，如果总共有T个step，那betas长度就是T
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0]) # 1000，若DDIM就是比如100

        # 这一部分用于前向加噪过程
        alphas = 1.0 - betas # (1000,)
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # alpha t的累乘 # (1000,)
        
        #### DDIM 相关设定
        if args.use_ddim:
            print(' === initializing DDIM ')
            timestep_respacing = args.timestep_respacing
            assert timestep_respacing != '',"Subseq Undefined"

            self.use_timesteps = set(space_timesteps(self.num_timesteps, timestep_respacing))
            self.timestep_map = [] #for indexing timestep value from subseq index

            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)

            self.betas = np.array(new_betas,dtype=np.float64)
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (self.betas > 0).all() and (self.betas <= 1).all()

            self.timestep_map = torch.tensor(self.timestep_map,dtype=torch.int64).cuda()
            self.num_timesteps = int(self.betas.shape[0])
            alphas = 1.0 - self.betas # (1000,)
            self.alphas_cumprod = np.cumprod(alphas, axis=0)
        print(f"\n modeltype: {self.modeltype}, diffusion step: {self.num_timesteps} \n")
        #### DDIM 相关设定

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # alpha t-1的累乘 # (1000,)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 这一部分用于反向去噪过程
        # calculations for diffusion q(xt | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # DDPM原文公式（4）的x0系数  根号(alpha的累乘)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 公式（3）的噪声系数
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | xt, x_0)
        
        # 2024-11-19 betas改self.betas 为支持DDIM
        self.posterior_variance = ( 
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 对应公式（7）中beta_t波浪
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # 2024-11-19 betas改self.betas 为支持DDIM
        self.posterior_mean_coef1 = ( 
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 公式（7）中后验均值 x_0的系数
        )
        self.posterior_mean_coef2 = ( # 公式（7）中后验均值 x_t的系数
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.schedule_sampler = create_named_schedule_sampler('uniform', self)

        self.predx0_list = []
        self.mean_list = []
        self.guide_list = []
        self.sample_list = []
        self.time_list = []


    def trainer_func_mdm(self, dataloader_iter, logger, optimizer, scheduler, optimizer_clip=None):
        ''' 跑新idea的 不属于CMC '''
        self.best_fid = 100
        self.best_Rprec = 0
        self.best_diff = 0 # Rprec-FID的差值
        if self.args.init_eval:
            FID, R_prec_top3 = self._eval_during_train(0)
            return
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            if self.args.dataset_name == 'snapmogen':
                clip_text, gt_motion, real_length = batch
            else:
                word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
                word_embeddings = word_embeddings.float().cuda()
                pos_one_hots = pos_one_hots.float().cuda()
                sent_len = sent_len.cuda()

            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.float().cuda()
            real_length = real_length.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)

            # save_path = os.path.join('visualization/', f'noisy_once_test_{nb_iter}.html')
            # vis_motion(pred_motion=gt_motion[0], gt_motion=None, save_path=save_path, vis=False)

            t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            
            if self.args.train_max_noisy_step != [0]:
                # ipdb.set_trace()
                assert len(self.args.train_max_noisy_step) == 2
                s_min = self.args.train_max_noisy_step[0]
                s_max = self.args.train_max_noisy_step[1]
                
                assert s_min<=s_max, f"s_min:{s_min}, s_max:{s_max}"
                if s_min == s_max:
                    t_init = torch.tensor(s_min).repeat(b).cuda()
                else:
                    t_init = torch.tensor(np.random.randint(s_min, s_max, size=b)).cuda()
                x0 = self.q_sample(gt_motion, t_init) 
            else:
                x0 = gt_motion

            if self.args.random_gt:
                x0 = torch.randn_like(x0).cuda()

            # 加噪
            noise = torch.randn_like(x0) 
            xt = self.q_sample(x0, t, noise=noise) 
            masked_xt = xt

            # 前向
            if self.modeltype in ['mdm', 'mdm_bert']:
                masked_xt = masked_xt.permute(0,2,1)[:,:,None]
                y={'text': clip_text}
                y['mask'] = real_mask
                
                # y['word_embs'] = word_embeddings
                # y['pos_ohot'] = pos_one_hots
                # y['cap_lens'] = sent_len
                pred_x0 = self.model(masked_xt, t, y=y)  # (b,196,263)
                pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
            elif self.modeltype == 'salad':
                pred_x0 = self.model(masked_xt, t, text=clip_text)  # (b,196,263)
            else:
                raise NotImplementedError

            if self.args.dataset_name == 'snapmogen':
                loss, msg = self._calc_mdm_loss_snapmogen(x0, pred_x0, real_length, real_mask, nb_iter, clip_text)
            else:
                loss, msg = self._calc_mdm_loss(x0, pred_x0, real_length, real_mask, nb_iter, word_embeddings, pos_one_hots, sent_len, optimizer, scheduler, optimizer_clip=optimizer_clip)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # if not self.args.ablation_separate_update:
            #     self._backward_and_step(loss, optimizer, scheduler)

            
            if nb_iter % self.args.print_iter == 0 :
                logger.info(msg)

            # self._save('test_lora.pth'); print(' debug =='*50)
            self._eval_and_save(nb_iter, logger)
            # if nb_iter % self.args.save_iter == 0 or (nb_iter > self.args.total_iter-100000 and nb_iter % 5000 == 0):
            #     if self.args.eval_during_train:
            #         fid, Rprec = self._eval_during_train(nb_iter)
            #         diff = Rprec-fid

            #         state_dict = self.model.state_dict()
            #         # 去掉text encoder的key
            #         clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            #         for e in clip_weights:
            #             del state_dict[e]

            #         if diff > best_diff:  
            #             best_diff = diff   
            #             torch.save(state_dict, pjoin(self.args.out_dir, 'net_best_diff.pth'))
            #             logger.info(f' save net_best_diff.pth') 



            #         if fid < best_fid:  
            #             best_fid = fid   
            #             torch.save(state_dict, pjoin(self.args.out_dir, 'net_best.pth'))
            #             logger.info(f' save net_best.pth') 
            #         if Rprec > best_Rprec:  
            #             best_Rprec = Rprec   
            #             logger.info(f' save net_bestR.pth') 
            #         logger.info(f'FID={fid}, best_FID={best_fid}')
            #         logger.info(f'diff={diff}, best_diff={best_diff}')
            #     logger.info('save net_last.pth')
            #     torch.save(state_dict, pjoin(self.args.out_dir, 'net_last.pth'))

    def _backward_and_step_ablation_separate_update(self, motion_loss, cos_loss, text_cos_loss, optimizer, scheduler, optimizer_clip):
        optimizer.zero_grad()
        motion_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        optimizer_clip.zero_grad()
        loss = self.args.cos_loss * cos_loss + self.args.text_cos_loss * text_cos_loss
        loss.backward()
        optimizer_clip.step()
        optimizer_clip.zero_grad()
        scheduler.step() # 还没成功


    def _backward_and_step(self, loss, optimizer, scheduler):
        optimizer.zero_grad()
        if self.args.adapt_el_tcl:
            self.optimizer_adapt.zero_grad()
        loss.backward()
        optimizer.step()
        if self.args.adapt_el_tcl:
            self.optimizer_adapt.step()
        scheduler.step()


    def _eval_and_save(self, nb_iter, logger):
        if nb_iter % self.args.save_iter == 0:
            if self.args.eval_during_train:
                fid, Rprec = self._eval_during_train(nb_iter)
                diff = Rprec-fid

                # state_dict = self.model.state_dict()
                # # 去掉text encoder的key
                # clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                # for e in clip_weights:
                #     del state_dict[e]

                if diff > self.best_diff:  
                    self.best_diff = diff   
                    self._save('net_best_diff.pth')
                    logger.info(f' save net_best_diff.pth') 


                if fid < self.best_fid:  
                    self.best_fid = fid   
                    filename = 'net_best.pth'
                    self._save(filename)
                    logger.info(f'save {filename}') 
                    if self.args.dataset_name == 'kit' and fid < 0.25:
                        filename = f'net_best_it{nb_iter}.pth'
                        self._save(filename)
                        logger.info(f'save {filename}') 

                    
                logger.info(f'FID={fid}, best_FID={self.best_fid}')
                logger.info(f'diff={diff}, best_diff={self.best_diff}')

            # logger.info('save net_last.pth')
            # torch.save(state_dict, pjoin(self.args.out_dir, 'net_last.pth'))
                
    def _save(self, ckpt_name):
        ckpt_path = pjoin(self.args.out_dir, ckpt_name)
        if self.args.add_clip_lora:
            state_dict = self.model.state_dict() # 全clip的key
            clip_all = {k: v for k, v in state_dict.items() if k.startswith('clip_model.')}
            mdm_base = {k: v for k, v in state_dict.items() if not k.startswith('clip_model.')}
            assert set(list(clip_all.keys())+list(mdm_base.keys())) == set(state_dict.keys())

            clip_lora = {k: v for k, v in state_dict.items() if 'lora' in k}

            dic = {}
            dic['base'] = mdm_base
            dic['clip_lora'] = clip_lora
            torch.save(dic, ckpt_path)
        else:
            torch.save(self.model.state_dict(), ckpt_path)
        print(f'Save ckpt to {ckpt_path}')

    def _eval_during_train(self, nb_iter):

        from eval_cmc import evaluation
        # 重新固定随机种子，确保测试过程的随机性可控
        fixseed(self.args.seed)
        
        self.model_cfg.eval()

        eval_motion_loaders = {
            ## HumanML3D Dataset##
            'vald': lambda: get_control_dataset(
                self.args, self.gen_loader, None, None, self, 0, 0, self.args.eval_sample_num
            )
        }
        if self.args.dataset_name == 'snapmogen':
            assert self.gen_loader.batch_size == 100, self.gen_loader.batch_size
        else:
            assert self.gen_loader.batch_size == 32, self.gen_loader.batch_size
        
        # 根据数据集类型选择 evaluator
        if self.args.dataset_name == 'snapmogen':
            # 使用 SnapMoGen 的 evaluator（本地模块）
            from models.snapmogen_evaluator import EvaluatorWrapper
            from utils.config_utils import load_config

            # 加载 SnapMoGen evaluator 配置
            eval_cfg = load_config('./SnapMoGen/checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml')
            eval_cfg.data.root_dir = '/data/motion/SnapMoGen'
            eval_cfg.exp.root_ckpt_dir = './SnapMoGen/checkpoint_dir'
            eval_wrapper = EvaluatorWrapper(eval_cfg, device=torch.device('cuda'))
            eval_wrapper.text_enc.eval()
            eval_wrapper.latent_enc.eval()

        else:
            if self.args.evaluator_eval is not None:
                if 'salad' in self.args.evaluator_eval:
                    opt = self.args
                    opt.device = 'cuda:0' # 自己加的
                    opt.latent_dim = 32
                    opt.activation = 'gelu'
                    opt.n_layers = 2
                    opt.n_extra_layers = 1
                    opt.kernel_size = 3 
                    opt.norm = 'none'
                    opt.dropout = 0.1
                    eval_wrapper = EvaluatorModelWrapperSALAD(opt, self.args.evaluator_eval)
                elif 'MARDM' in self.args.evaluator_eval:
                    eval_wrapper = EvaluatorMARDM(self.args.dataset_name, torch.device('cuda'))
                else:
                    eval_wrapper = EvaluatorMDMWrapper(self.args.dataset_name, torch.device('cuda'), self.args, self.args.evaluator_eval)
            else:
                eval_wrapper = EvaluatorMDMWrapper(self.args.dataset_name, torch.device('cuda'), self.args, self.args.evaluator_eval)

            eval_wrapper.text_encoder.eval()
            eval_wrapper.motion_encoder.eval()
            eval_wrapper.movement_encoder.eval() 

        # ['Matching Score_ground truth', 'Matching Score_vald', 'R_precision_ground truth', 'R_precision_vald', 'FID_ground truth', 'FID_vald', 'Diversity_ground truth', 'Diversity_vald']
        metric_dict = evaluation(eval_wrapper, self.gt_loader, eval_motion_loaders, self.log_file)
        
        # 测试结束后再次固定随机种子，确保后续训练的连续性和可重复性
        fixseed(self.args.seed)
        
        self.model_cfg.train()
        FID = metric_dict['FID_vald']
        R_prec_top3 = metric_dict['R_precision_vald'][2]
        # skating_ratio = metric_dict['Skating Ratio_vald']
        self.writer.add_scalar('Metric/FID', FID, nb_iter)
        self.writer.add_scalar('Metric/R_prec_top3', R_prec_top3, nb_iter)
        # self.writer.add_scalar('Metric/Skating_Ratio', skating_ratio, nb_iter)
        return FID, R_prec_top3

    def _calc_mdm_loss_snapmogen(self, gt, pred, real_length, real_mask, iter, clip_text):
        loss = 0
        B, L, dim = gt.shape
        msg = f' Train. Iter {iter} '


        # 1. 运动重建损失 (MSE)
        motion_real_mask = real_mask[..., None].repeat(1, 1, dim)
        motion_loss = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
        msg += f" motion_loss. {motion_loss:.5f}"
        self.writer.add_scalar('Loss/motion_loss', motion_loss.item(), iter)
        loss += motion_loss

        if self.args.text_cos_loss:
            # evaluator 的输入维度（148），训练数据维度为 296
            eval_dim = self.eval_wrapper.latent_enc.nfeats
            text_emb, _ = self.eval_wrapper.encode_text(clip_text, sample_mean=True)  # (batch, latent_dim)
            text_emb = text_emb.to(gt.device)
            with torch.enable_grad():
                _, pred_emb, _ = self.eval_wrapper.encode_motion(pred[..., :eval_dim], real_length, sample_mean=False)

            text_cos_loss = 1 - F.cosine_similarity(text_emb, pred_emb, dim=-1).mean()
            self.writer.add_scalar('Loss/text_cos_loss', text_cos_loss.item(), iter)
            loss += self.args.text_cos_loss * text_cos_loss
            msg += f" text_cos_loss. {text_cos_loss:.4f}"

        if self.args.cos_loss:
            _, gt_emb, _ = self.eval_wrapper.encode_motion(gt[..., :eval_dim], real_length, sample_mean=False)
            with torch.enable_grad():
                _, pred_emb, _ = self.eval_wrapper.encode_motion(pred[..., :eval_dim], real_length, sample_mean=False)

            # 计算余弦相似度损失
            cos_loss = 1 - F.cosine_similarity(gt_emb, pred_emb, dim=-1).mean()
            self.writer.add_scalar('Loss/cos_loss', cos_loss.item(), iter)
            loss += self.args.cos_loss * cos_loss
            msg += f" cos_loss. {cos_loss:.5f}"

        return loss, msg

    
    def _calc_mdm_loss(self, gt, pred, real_length, real_mask, iter, word_embeddings, pos_one_hots, sent_len, optimizer, scheduler, optimizer_clip=None):
        loss = 0
        B,L,dim = gt.shape
        msg = f' Train. Iter {iter} '

        unnormed_pred = pred * self.std[..., :dim] + self.mean[..., :dim]
        unnormed_gt = gt * self.std[..., :dim] + self.mean[..., :dim]

        motion_real_mask = real_mask[..., None].repeat(1,1, dim)
        motion_loss = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
        msg += f" motion_loss. {motion_loss:.5f}"
        self.writer.add_scalar('Loss/motion_loss', motion_loss.item(), iter)

        loss += motion_loss

        if self.args.bone_loss:
            bone_loss = self._calc_bone_loss(unnormed_pred)
            loss += self.args.bone_loss * bone_loss
            msg += f" bone_loss. {bone_loss:.4f}"
            self.writer.add_scalar('Loss/bone_loss', bone_loss.item(), iter)

        if self.args.ric_global_loss:
            gt_joint = recover_from_ric(unnormed_gt, self.n_joints) # (b,196,22,3)
            pred_joint = recover_from_ric(unnormed_pred, self.n_joints)
            if self.args.dataset_name == 'kit':
                gt_joint /= 1000
                pred_joint /= 1000
            joint_mask = real_mask[..., None, None].repeat(1, 1, self.n_joints, 1)
            ric_loss = mean_joint_error_torch(gt_joint, pred_joint, joint_mask)
            loss += self.args.ric_global_loss * ric_loss
            msg += f" ric_global_loss. {ric_loss:.5f}"

        if self.args.ric_rot_loss:
            ric_rot_loss = self._calc_ric_rot_loss(unnormed_pred, real_mask)
            # loss += self.args.ric_rot_loss * ric_rot_loss
            msg += f" ric_rot. {ric_rot_loss:.5f}"

        if self.args.text_emb_loss or self.args.text_cos_loss or self.args.text_infonce_loss:
            
            text_emb, pred_emb = self.eval_wrapper.get_co_embeddings_with_grad(word_embeddings, pos_one_hots, sent_len, pred, real_length)
            text_emb, gt_emb = self.eval_wrapper.get_co_embeddings_with_grad(word_embeddings, pos_one_hots, sent_len, gt, real_length)
            
            if self.args.text_emb_loss:
                text_emb_loss = F.mse_loss(text_emb, pred_emb)
                self.writer.add_scalar('Loss/text_emb_loss', text_emb_loss.item(), iter)
                loss += self.args.text_emb_loss * text_emb_loss
                msg += f" text_emb_loss. {text_emb_loss:.4f}"
            elif self.args.text_cos_loss:
                text_cos_loss = 1 - F.cosine_similarity(text_emb, pred_emb, dim=-1).mean()
                self.writer.add_scalar('Loss/text_cos_loss', text_cos_loss.item(), iter)
                loss += self.args.text_cos_loss * text_cos_loss
                msg += f" text_cos_loss. {text_cos_loss:.4f}"

        
        if self.args.emb_loss or self.args.cos_loss or self.args.infonce_loss or self.args.dist_loss:
            if not (self.args.text_emb_loss or self.args.text_cos_loss or self.args.text_infonce_loss):
                # gt_emb = self.eval_wrapper.get_motion_embeddings(motions=gt, m_lens=real_length)
                gt_emb = self.eval_wrapper.get_motion_embeddings_with_grad(motions=gt, m_lens=real_length) # 加SnapMoGen的代码时改成这个了
                pred_emb = self.eval_wrapper.get_motion_embeddings_with_grad(motions=pred, m_lens=real_length)
            
            if self.args.emb_loss:
                emb_loss = F.mse_loss(gt_emb, pred_emb)
                self.writer.add_scalar('Loss/emb_loss', emb_loss.item(), iter)
                loss += self.args.emb_loss * emb_loss
                msg += f" emb_loss. {emb_loss:.5f}"
            elif self.args.cos_loss:
                cos_loss = 1 - F.cosine_similarity(gt_emb, pred_emb, dim=-1).mean()
                self.writer.add_scalar('Loss/cos_loss', cos_loss.item(), iter)
                loss += self.args.cos_loss * cos_loss
                msg += f" cos_loss. {cos_loss:.5f}"
            elif self.args.dist_loss:
                dist_loss = F.pairwise_distance(gt_emb, pred_emb, p=2).mean()
                self.writer.add_scalar('Loss/dist_loss', dist_loss.item(), iter)
                loss += self.args.dist_loss * dist_loss
                msg += f" dist_loss. {dist_loss:.5f}"



        if self.args.l_smooth:
            l_smooth = pred[:, 1:, :] - pred[:, :-1, :]
            l_smooth = l_smooth.pow(2).mean()
            loss += self.args.l_smooth * l_smooth
            msg += f" l_smooth. {l_smooth:.5f}"
        
        # if self.args.ablation_separate_update:
        #     self._backward_and_step_ablation_separate_update(motion_loss, cos_loss, text_cos_loss, optimizer, scheduler, optimizer_clip)



        return loss, msg
    
    def _calc_ric_rot_loss(self, pred, real_mask, return_dist=True, need_denorm=False):
        '''
        默认return_dist用距离的loss
        '''
        if need_denorm:
            pred = pred * self.std + self.mean
        b = pred.shape[0]
        joint_mask = real_mask[..., None, None].repeat(1, 1, self.n_joints, 3) # (b,196,22,3)
        ric_pos = recover_from_ric(pred, joints_num=self.n_joints)
        rot_pos = recover_from_rotation(pred, joints_num=self.n_joints).reshape(b,196,self.n_joints,3)
        
        if not return_dist:
            ric_rot_loss = F.mse_loss(ric_pos[joint_mask], rot_pos[joint_mask])
            return ric_rot_loss
            # vis_motion(pred[0], pred[0], joint2_from_joint1rot=True, vis=True, need_denorm=False)
        else:
            mean_error = mean_joint_error_torch(ric_pos, rot_pos, joint_mask[..., 0:1])
            return mean_error
    
    def _calc_bone_loss(self, pred):
        pred_joints = recover_from_ric(pred, joints_num=self.n_joints)[:,:40,:,:].flatten(0,1)
        
        # # 示例数据
        # B, T, J, C = 4, 196, 22, 3
        # motion = torch.randn(B, T, J, C)             # shape: (B, 196, 22, 3)
        # lengths = torch.tensor([120, 90, 196, 150])  # shape: (B,)

        # # Step 1: 生成 mask，标出哪些帧是有效的
        # device = motion.device
        # lengths = lengths.to(device)
        # frame_ids = torch.arange(T, device=device).unsqueeze(0)  # shape: (1, T)
        # valid_mask = frame_ids < lengths.unsqueeze(1)            # shape: (B, T), bool

        # # Step 2: 展平 motion 到 (B*T, J, C)，mask 到 (B*T,)
        # motion_flat = motion.view(-1, J, C)                      # shape: (B*T, J, C)
        # valid_mask_flat = valid_mask.view(-1)                   # shape: (B*T,)

        # # Step 3: 用 mask 选出有效帧
        # motion_valid = motion_flat[valid_mask_flat]             # shape: (N, 22, 3)

        if self.args.dataset_name == 't2m':
            gt_offset = t2m_tgt_offsets[None, ...].repeat(self.args.batch_size*40, 1, 1).to(pred.device)
            pred_offset = t2m_tgt_skel.calc_offsets_joints_batch(pred_joints)
            bone_loss = F.l1_loss(pred_offset, gt_offset)
        elif self.args.dataset_name == 'kit':
            gt_offset = kit_tgt_offsets
            pred_offset = kit_tgt_skel.calc_offsets_batch(pred_joints)
            bone_loss = F.l1_loss(pred_offset, gt_offset)
        return bone_loss

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    @torch.no_grad()
    def p_sample_loop(self, partial_emb, with_control=True, model_kwargs=None, batch_size=1, indices_bound=None):
        '''
        partial_emb: (b,196,263)
        condition: 字典，包含文本条件和轨迹条件
        '''
        if sys.gettrace():
            self.predx0_list = []
            self.mean_list = []
            self.guide_list = []

        B = batch_size # batch_size
        skip_t = 0
        if indices_bound is not None:
            assert isinstance(indices_bound, list)
            assert len(indices_bound) == 2 and indices_bound[0] > indices_bound[1]
            indices = list(range(indices_bound[0], indices_bound[1]-1, -1))
        else:
            indices = list(range(self.num_timesteps - skip_t))[::-1]

        if self.args.dataset_name == 't2m':
            motion_dim = 263
            ric_dim = 67
            seq_len = 196
        elif self.args.dataset_name == 'kit':
            motion_dim = 251
            ric_dim = 64
            seq_len = 196
        elif self.args.dataset_name == 'snapmogen':
            motion_dim = 296
            ric_dim = 73  # 24 joints * 3 - 1 (root) = 71, but SnapMoGen uses 296 dim
            seq_len = self.args.max_motion_length  # 320 for SnapMoGen
        else:
            raise NotImplementedError

        if self.modeltype in ['mdm','salad', 'mdm_bert']:
            noise = torch.randn((B, seq_len, motion_dim)).cuda()
        else:
            print('self.modeltype = ', self.modeltype)
            raise ValueError('Unknown model type')

        xt = noise
        with torch.no_grad():
            for i in tqdm(indices): # 999 ~ 0
                t = torch.tensor([i] * B).cuda() # timestep tensor
                if self.args.use_ddim:
                    out = self.ddim_sample(xt, t, partial_emb, model_kwargs=model_kwargs)
                else:
                    out = self.p_sample(xt, t, partial_emb, model_kwargs=model_kwargs) # 返回x_{t-1}和x0
                xt = out["sample"] # x_{t-1}
        
        # if sys.gettrace():
        #     self.plot_xyz_error(model_kwargs['traj'])

        # 只有CMC的2阶段会做替换
        if self.modeltype in ['diffmdm'] and with_control and partial_emb is not None: 
            out['sample'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['sample']) 
        return out['sample']

    def p_sample(self, xt, t, partial_emb, model_kwargs=None):
        ''' get x_{t-1}
        '''
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs) 

        if self.args.gen_with_text_grad:
            # ipdb.set_trace()
            out['mean'] = self._guide_text_grad(out['mean'], model_kwargs)

        mean = out['mean']   
        var = out['variance']
        log_var = out['log_variance']
        pred_x0 = out['pred_x0']

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1))) # no noise when t == 0
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise 
            

        return {"sample": sample,
                "pred_x0": pred_x0} # 分别是x_{t-1}和x_0
    
    def ddim_sample(self, xt, t, partial_emb, model_kwargs=None, eta=0.5):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs)      

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(xt, t, out["pred_x0"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, xt.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, xt.shape)
        sigma = (eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = torch.randn_like(xt)
        # 标准的DDIM到这就没有了，这里就是计算前一个跨步的step, 即xs。上面算sigma只是为了增加一点点随机性。这里的mean_pred和DDPM中的posterior_mean是不一样的
        mean_pred = (
            out["pred_x0"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise


        return {"sample": sample,
                "pred_x0": out['pred_x0']}


    def p_mean_variance(self, masked_xt, t, model_kwargs=None):
        ''' get pred_x0
        '''
        B = masked_xt.shape[0]
        assert t.shape == (B,)
        
        
        assert masked_xt.shape[0] == len(model_kwargs['clip_text'])
        if self.args.use_ddim:   # 2024-11-20添加，用于DDIM
            sample_t = self.timestep_map[t]
        else:
            sample_t = t.clone()
        # masked_xt = torch.ones_like(masked_xt, device=masked_xt.device) * 0.5; print(' for debug')
        # 前向推理
        if self.modeltype in ['diffmdm', 'mdm', 'mdm_bert']:
            xt = masked_xt.permute(0,2,1)[:,:,None]
            y = model_kwargs
            y['text'] = model_kwargs['clip_text']
            y['mask'] = model_kwargs['real_mask']
            if y.get('text_emb') is not None:
                y['text_emb'] = y.get('text_emb')
            pred_x0 = self.model_cfg(xt, sample_t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
        elif self.modeltype in ['salad']:
            xt = masked_xt
            y={'text': model_kwargs['clip_text']}
            pred_x0 = self.model_cfg(xt, sample_t, y=y)  # (b,196,263)
        else:
            raise NotImplementedError
        

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, masked_xt.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, masked_xt.shape)

        # 得到x0后去算x_{t-1}的均值，即后验均值 q(x_{t-1} | x_t, x_0)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=masked_xt, t=t) 

        assert model_mean.shape == model_log_variance.shape == pred_x0.shape == masked_xt.shape
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }
    
    
    

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        DDPM原论文公式(7) q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = ( # 公式（7）中的mu_t就是这个后验均值
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def _guide_text_grad(self, x, model_kwargs):

        lr = 0.5 
         
        with torch.enable_grad():
            x = x.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                word_embs = model_kwargs['word_embs']
                pos_ohot = model_kwargs['pos_ohot']
                cap_lens = model_kwargs['cap_lens']
                real_length = model_kwargs['real_length']
                
                text_emb, motion_emb = self.eval_wrapper.get_co_embeddings_with_grad(word_embs, pos_ohot, cap_lens, x, real_length)
                loss = F.mse_loss(text_emb, motion_emb)
                loss.backward()
                return loss

            lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=50,
                        lr = lr,
                        tolerance_change=1e-8,
                        line_search_fn="strong_wolfe")
                
            lbfgs.step(closure)
        return x
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)