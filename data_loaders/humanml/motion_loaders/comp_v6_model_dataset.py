import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
from utils.mask_utils import generate_src_mask, calc_loss_xyz, calc_loss_xyz_perbatch, vis_motion
from utils.fixseed import fixseed

class CompADCGeneratedDataset(Dataset):
    def __init__(self, args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit):
        if args.seed:
            fixseed(args.seed)
        self.args = args
        self.gen_loader = gen_loader
        self.dataset = gen_loader.dataset
        assert mm_num_samples < len(gen_loader.dataset)
        num_samples_limit = len(self.dataset) if num_samples_limit > len(self.dataset) else num_samples_limit
        real_num_batches = len(gen_loader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // gen_loader.batch_size + 1
        print('real_num_batches', real_num_batches)


        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // gen_loader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs = ', mm_idxs)
        # samples = []
        # gt_motions = []
        
        for i, batch in enumerate(self.gen_loader):
            print(f'{i}/{real_num_batches}')
            if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                break
            
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            txt_tokens = [t.split('_') for t in txt_tokens]
            b, max_length, num_features = gt_motion.shape
            word_embeddings = word_embeddings.float().cuda()
            pos_one_hots = pos_one_hots.float().cuda()
            sent_len = sent_len.cuda()
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask = traj_mask.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
    


            model_kwargs = {}
            # model_kwargs['text_emb'] = text_emb
            # model_kwargs['word_emb'] = word_emb
            model_kwargs['gt_motion'] = gt_motion
            model_kwargs['real_mask'] = real_mask
            model_kwargs['clip_text'] = clip_text
            model_kwargs['word_embs'] = word_embeddings
            model_kwargs['pos_ohot'] = pos_one_hots
            model_kwargs['cap_lens'] = sent_len
            model_kwargs['real_length'] = real_length

            # ipdb.set_trace()


            is_mm = i in mm_idxs 
            repeat_times = mm_num_repeats if is_mm else 1
            mm_motions = []

        
            for t in range(repeat_times):
                if self.args.test_gt_metric:
                    sample = gt_motion
                else:
                    partial_emb = None
                    # test_with_noisy_step默认是0， 这是用来测试对GT进行 最大步数为50的 2、4、6、8、10步加噪的指标的；不是测生成指标
                    if args.test_with_noisy_step:
                        t_init = torch.tensor(args.test_with_noisy_step).repeat(b).cuda()
                        sample = diffusion.q_sample(gt_motion, t_init)
                    else:
                        sample = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=32) # (b, 196, 263)

                # vis_motion(motion1=sample[0], motion2=gt_motion[0], save_path='visualization/1.html', vis=True)

                if t == 0:
                    sub_dicts = [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                'length': real_length[bs_i].cpu().numpy(),
                                'caption': clip_text[bs_i],
                                'hint': traj[bs_i].cpu().numpy(),
                                'tokens': txt_tokens[bs_i],
                                'cap_len': sent_len[bs_i].item(),
                                'filename': filename[bs_i],
                                } for bs_i in range(gen_loader.batch_size)]
                    
                    generated_motion += sub_dicts
                    
                if is_mm:
                    mm_motions += [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                    'length': real_length[bs_i].cpu().numpy(),
                                    } for bs_i in range(gen_loader.batch_size)]

            
            if is_mm:
                mm_generated_motions += [{
                                'caption': clip_text[bs_i],
                                'tokens': txt_tokens[bs_i],
                                'cap_len': sent_len[bs_i].item(),
                                'mm_motions': mm_motions[bs_i::gen_loader.batch_size], 
                                } for bs_i in range(gen_loader.batch_size)]
            a = 1

        print('last sample.sum()=', sample.sum())
        
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = gen_loader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, hint, filename = data['motion'], data['length'], data['caption'], data['tokens'], data['hint'], data['filename']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            # vis_motion(motion1=denormed_motion, save_path='visualization/1.html', vis=True)
            dim = self.dataset.mean_for_eval.shape[0]
            renormed_motion = (denormed_motion[:, :dim] - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms  适配MARDM的dim67的评估器
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint, filename
    


class CompSnapMoGen(Dataset):
    def __init__(self, args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit):
        if args.seed:
            fixseed(args.seed)
        self.args = args
        self.gen_loader = gen_loader
        self.dataset = gen_loader.dataset
        assert mm_num_samples < len(gen_loader.dataset)
        num_samples_limit = len(self.dataset) if num_samples_limit > len(self.dataset) else num_samples_limit
        real_num_batches = len(gen_loader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // gen_loader.batch_size + 1
        print('real_num_batches', real_num_batches)


        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // gen_loader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs = ', mm_idxs)
        
        for i, batch in enumerate(self.gen_loader):
            # if i==10:
            #     break
            print(f'{i}/{real_num_batches}')
            if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                break
            
            clip_text, gt_motion, real_length = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
    

            model_kwargs = {}
            model_kwargs['real_mask'] = real_mask
            model_kwargs['clip_text'] = clip_text



            is_mm = i in mm_idxs 
            repeat_times = mm_num_repeats if is_mm else 1
            mm_motions = []

        
            for t in range(repeat_times):
                if self.args.test_gt_metric:
                    sample = gt_motion
                else:
                    partial_emb = None
                    # test_with_noisy_step默认是0， 这是用来测试对GT进行 最大步数为50的 2、4、6、8、10步加噪的指标的；不是测生成指标
                    if args.test_with_noisy_step:
                        t_init = torch.tensor(args.test_with_noisy_step).repeat(b).cuda()
                        sample = diffusion.q_sample(gt_motion, t_init)
                    else:
                        sample = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=32) # (b, 196, 263)

                # vis_motion(motion1=sample[0], motion2=gt_motion[0], save_path='visualization/1.html', vis=True)

                if t == 0:
                    sub_dicts = [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                'length': real_length[bs_i].cpu().numpy(),
                                'caption': clip_text[bs_i],
                                } for bs_i in range(gen_loader.batch_size)]
                    
                    generated_motion += sub_dicts
                    
                if is_mm:
                    mm_motions += [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                    'length': real_length[bs_i].cpu().numpy(),
                                    } for bs_i in range(gen_loader.batch_size)]

            
            if is_mm:
                mm_generated_motions += [{
                                'caption': clip_text[bs_i],
                                'mm_motions': mm_motions[bs_i::gen_loader.batch_size], 
                                } for bs_i in range(gen_loader.batch_size)]
            a = 1

        print('last sample.sum()=', sample.sum())
        
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions


    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption= data['motion'], data['length'], data['caption']
        return caption, motion, m_length
    