import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from dataset.snapmogen_dataset import TextMotionDataset as SnapMoGenTextMotionDataset
from torch.utils.data._utils.collate import default_collate
from utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from utils.motion_process import recover_root_rot_pos, recover_from_ric, recover_from_rot
from utils.config_utils import load_config
import sys

from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import ipdb
from utils.mask_utils import vis_motion

def create_trajmask263(joint_ids, frames=None, dataset_name='t2m', mode='train'):
    """ create trajectory mask for motion representation in HumanML3D/KIT for DiffMoAE

    Args:
        joint_ids (np.ndarray): 
        frames (np.ndarray):
    Returns:
        traj_mask: (L, 22, 3)    for calculating global xyz loss
        traj_mask_263: (L, 263)  for DiffMoAE
    """
    assert isinstance(joint_ids, np.ndarray)
    if frames is None:
        frames = np.arange(L)
    else:
        assert isinstance(frames, np.ndarray)

    L = 196

    if dataset_name == 't2m':
        traj_mask = np.zeros((L, 22, 3)).astype(bool)
        traj_mask_263 = np.zeros((L, 263)).astype(bool)
    elif dataset_name == 'kit':
        traj_mask = np.zeros((L, 21, 3)).astype(bool)
        traj_mask_263 = np.zeros((L, 251)).astype(bool)
    else:
        raise NotImplementedError(f'{dataset_name} not supported')

    traj_mask_263[:, :4] = True # root
    for i in joint_ids:
        traj_mask[frames, i] = True
        traj_mask_263[frames, 4+3*(i-1):4+3*i] = True # ric  21*3

    return traj_mask, traj_mask_263

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class ControlDataset(data.Dataset):
    def __init__(self, opt, args, mean, std, split_file, w_vectorizer, mode, 
                 control_joint=0, density=100, dataset_name='t2m', normalize_traj=False, 
                 multi_joint_control=False, unit_length=None, codebook_dir=None, diffusion=None):
        
        self.opt = opt
        self.args = args
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.split_file = split_file
        self.mode = mode
        self.control_joint = control_joint
        self.density = density
        self.normalize_traj = normalize_traj
        self.multi_joint_control = multi_joint_control
        self.diffusion = diffusion
        # print('self.control_joint = ', self.control_joint)
        # print('self.density = ', self.density)
        assert  0 <= self.density <= 100, "density should be in [0, 100], got {}".format(self.density)

        # 以下是VQ相关的
        self.unit_length = unit_length # 1个token对应实际的动作长度，如果VQVAE里下采样2次，那么unit_length=4
        # self.codebook_dir = codebook_dir # 训练前预存的motion tokens的路径 output/vq/vq_name/codebook
        # self.codebook_size = codebook_size
        # self.motion_end_idx = codebook_size
        # self.motion_pad_idx = codebook_size + 1
        self.max_token_length = 26 if unit_length == 8 else 50
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        

        fps = 20 if self.opt.dataset_name == 't2m' else 12.5 # HumanML3D帧率20; KIT帧率12.5

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        
        if self.args.train_sample_num and 'train' in split_file:
            id_list = id_list[:self.args.train_sample_num]
        # id_list = id_list[:333]; print(' !!!!!!!!!!!!!!!!!! debug \n !!!!!!!!!!!!!!!!!! debug \n !!!!!!!!!!!!!!!!!! debug \n !!!!!!!!!!!!!!!!!! debug') # debug用
        if args.no_random:
            id_list = id_list[:111] 
        # if self.mode == 'debug':
        if sys.gettrace():
            id_list = id_list[:111]

        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # if self.codebook_dir:
                #     motion_token =  np.load(pjoin(self.codebook_dir, name + '.npy'))[0] # 因为读进来是(1,L) 实际上只有1个
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                # 这部分的意思是，一段动作，但是有的文本对应的动作段是有起始点和结束点的
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag  # 起始秒 from_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag # 结束秒 to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps) : int(to_tag * fps)] # 起始秒和结束秒 乘上帧率 KIT帧率是12.5
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200): # 过滤长度过短过长的动作
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                                # if self.codebook_dir:
                                #     motion_token_ = motion_token[int(f_tag * fps / unit_length) : int(to_tag * fps / unit_length)]
                                #     data_dict[new_name]['motion_token'] = motion_token_

                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    # if self.codebook_dir:
                    #     data_dict[name]['motion_token'] = motion_token
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])) # 把名字长度二元组依长度升序排列
        name_list = new_name_list # debug用
        # length_list = length_list # debug用

        self.mean = mean
        self.std = std
        self.dim = mean.shape[0]

        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = './dataset/humanml_spatial_norm'
            n_joints = 22
        elif 'KIT' in opt.data_root:
            spatial_norm_path = './dataset/kit_spatial_norm'
            n_joints = 21
        else:
            raise NotImplementedError('unknown dataset')
        # 全局xyz的均值和方差；原本HumanML3D的Mean.npy是相对xyz的均值和方差
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy')).reshape(n_joints,3)
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy')).reshape(n_joints,3)
        
        self.data_dict = data_dict
        self.name_list = name_list
        print(f'=== total {len(self.data_dict)} data')
        
        # 对数据一次性加噪再去训练
        # if self.args.init_noisy_data_level and 'train' in split_file:
        #     for i, k in enumerate(self.data_dict.keys()):
        #         t_init = torch.tensor([self.args.init_noisy_data_level])
        #         motion = self.data_dict[k]['motion']
        #         normed_motion = self.transform(motion)[None,:]
        #         noisy_normed_motion = self.diffusion.q_sample(torch.tensor(normed_motion), t_init)
        #         assert normed_motion.sum() != noisy_normed_motion.sum()
        #         self.data_dict[k]['motion'] = self.inv_transform(noisy_normed_motion).numpy()[0]

        #         if i<5:
        #             save_path = os.path.join(f'{self.args.out_dir}', f'noisy_once_test_{i}.html')
        #             vis_motion(pred_motion=noisy_normed_motion[0], gt_motion=normed_motion[0], save_path=save_path, vis=False)
        
    
    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean
    

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
            joints_name = np.array(['pelvis', 'left_foot', 'right_foot', 'head', 'left_wrist', 'right_wrist'])
        elif n_joints == 21:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF'] # 根，头，左手，右手，左脚，右脚
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        else:
            raise NotImplementedError

        # 选择控制关节
        # num_joints = len(controllable_joints)
        # if self.multi_joint_control:
        #     num_joints_control = np.random.choice(np.arange(1, num_joints+1), 1) # 1~6  多关节控制
        # else:
        #     num_joints_control = 1
        # choose_joint = np.random.choice(num_joints, num_joints_control, replace=False) # 选择控制的关节点
        # # choose_name = joints_name[choose_joint]
        # choose_joint = controllable_joints[choose_joint]
        
        if isinstance(self.control_joint, list):
            if self.control_joint == [-1]: # default -1, 随机选取控制关节数
                num_joints = len(controllable_joints)
                if self.multi_joint_control:
                    num_joints_control = np.random.choice(np.arange(1, num_joints+1), 1) # 1~6  多关节控制
                else:
                    num_joints_control = 1
                choose_joint = np.random.choice(num_joints, num_joints_control, replace=False) # 选择控制的关节点
                choose_joint = controllable_joints[choose_joint]
            else:
                choose_joint = np.array(self.control_joint)
        else:
            pass

        # print(choose_joint)
        assert set(choose_joint).issubset(controllable_joints), choose_joint



        # 选择控制帧比例
        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1 # 随机设定控制的帧数 范围 [1,L-1]
        if self.density:
            if self.density in [1, 2, 5]:
                choose_seq_num = self.density
            else:
                choose_seq_num = int(length * self.density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False) # 根据帧数选择控制的时刻帧
        choose_seq.sort()
        # print('frames percent = ', choose_seq_num/length)

        traj_mask, traj_mask_263 = create_trajmask263(choose_joint, choose_seq, dataset_name=self.opt.dataset_name, mode=self.mode)

        # normalize
        if self.normalize_traj: # omnicontrol最原本就是不归一化轨迹
            joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * traj_mask[:length, ...]
        return joints, traj_mask_263, traj_mask
    


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        '''随机性
        1. 文本随机 text_data = random.choice(text_list)
        2. coin2 = np.random.choice(['single', 'single', 'double'])
        3. 动作随机起点截取 idx = random.randint(0, len(motion) - m_length)
        '''
        idx = self.pointer + item
        # idx = 29
        # idx = 1316
        # idx = 29; print(f' idx={idx} , {self.name_list[idx]} for debug')
        # idx = 120; print(f' idx={idx} for debug') # 站着挥右手
        # idx = 179; print(f' idx={idx}for debug') # 站着挥双手,错误人体，正反面反过来的
        filename = self.name_list[idx]
        # filename = '009613'; print(f' filename = {filename} for debug')
        # filename = '004822'; print(f' filename = {filename} for debug') # example1
        # filename = '002662'; print(f' filename = {filename} for debug')
        # filename = '008382'; print(f' filename = {filename} for debug')
        data = self.data_dict[filename]

        # data = self.data_dict['000007'] 
        # if idx <= 32:
        
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # motion = np.load(f'dataset/HumanML3D/new_joint_vecs/{self.name_list[idx]}.npy'); print('for debug !!!')
        # m_length = 199; print('for debug !!!')

        # Randomly select a caption
        text_data = random.choice(text_list)
        if self.args.no_random:
            text_data = text_list[0]; print('choose 0th text, for debug !!!') # ① 固定文本
        caption, tokens = text_data['caption'], text_data['tokens']

        

        if len(tokens) < self.opt.max_text_len:
            # 句子短，补SOS和EOS token，然后补unknown token至固定长度
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # 句子场，固定切割到固定长度，再补SOS EOS
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)        # (22,15)   22是预设的句子最大长度20加上2个SOS和EOS
        word_embeddings = np.concatenate(word_embeddings, axis=0)  # (22,300)

        # 将动作长度截取为unit_length即4的整数倍，并通过coin2引入一些随机，无需细抠
        # if self.mode == 'train':
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if self.args.no_random:
            coin2 = 'single' # ② 固定coin
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        # else:
        #     m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        i = random.randint(0, len(motion) - m_length)
        if self.args.no_random:
            i = 0 # ③ 固定初始帧
        motion = motion[i:i+m_length]

        n_joints = 22 if motion.shape[-1] == 263 else 21
        # hint is global position of the controllable joints
        joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints) # (L, 22, 3)  每个关节点的全局坐标
        joints = joints.numpy()
        ##########################
        # joints_disk = np.load(f'dataset/HumanML3D/new_joints/{self.name_list[idx]}.npy')
        # assert np.allclose(joints, joints_disk[:196])

        # tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
        # tgt_skel.get_offsets_joints(torch.from_numpy(joints[0]))
        # joints_rot1 = recover_from_rot(torch.from_numpy(motion), n_joints, tgt_skel).numpy()
        # assert np.allclose(joints, joints_rot1, atol=1e-6)

        # example_data = np.load('/data/motion/HumanML3D/new_joints/000021.npy')
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
        # joints_rot2 = recover_from_rot(torch.from_numpy(motion), n_joints, tgt_skel).numpy()
        # assert np.allclose(joints, joints_rot2, atol=1e-6)
        ##########################``
        # control any joints at any time
        hint, traj_mask_263, traj_mask = self.random_mask_train(joints, n_joints) # joints: (L,22,3) 从开头到这都查过了与omnicontrol一致没错


        hint = hint.reshape(hint.shape[0], -1) # (L,22*3)

        # motion 263的归一化
        motion = (motion[:, :self.dim] - self.mean) / self.std
        
        

        if m_length < self.max_motion_length: # 固定输出动作长度为max_length ！！
            hint   = np.concatenate([hint, np.zeros((self.max_motion_length - m_length, hint.shape[1])) ], axis=0)
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1])) ], axis=0)

        hint = hint.astype(np.float32).reshape(self.max_motion_length, n_joints, 3)
        motion = motion.astype(np.float32)
        

        # 确保取得的轨迹以及traj_mask正确
        if self.normalize_traj:
            # joints: L,22,3
            # hint: 196,22,3
            assert np.allclose(joints * traj_mask[:m_length, ...] , ((hint * self.raw_std + self.raw_mean) * traj_mask)[:m_length, ...], atol=1e-4) # HumanML3D这里阈值可以是1e-6，kit只能是1e-4
        else:
            assert (joints*traj_mask[:m_length, ...] - hint[:m_length, ...]).sum() == 0

        if self.args.no_random and filename == '004822':
            print('name = ', self.name_list[idx], 'motion.sum = ', motion.sum())
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint, traj_mask_263, traj_mask, filename
        

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, args, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, normalize_traj=False, multi_joint_control=False, unit_length=None, density=100, diffusion=None, **kwargs):
        self.mode = mode
        self.split = split
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir) # ./dataset/HumanML3D/new_joint_vecs
        opt.text_dir = pjoin(abs_base_path, opt.text_dir) # ./dataset/HumanML3D/texts
        opt.data_root = pjoin(abs_base_path, opt.data_root) # ./dataset/HumanML3D/
        opt.meta_dir = './dataset'
        self.opt = opt
        self.dataset_name = opt.dataset_name
        print('Loading dataset %s ...' % opt.dataset_name)


        '''
        meta_mean不适用于MDM, 因为有一个方差很小导致归一化之后特别大, 导致收敛困难; 可用于训练评估器或者VQ-based的模型
        Mean是数据集直接计算出来的, 可以直接用于训练任何模型, 只是以前VQ用的都是meta_mean的就直接跟了。
        '''

        if mode == 'gt':
            # used by T2M models (including evaluators) 这就是MDM里的t2m_mean.npy 即 t2m/Comp_v6_KLD005/meta/mean.npy
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            if args.using_meta:
                self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy')) # sum = 49.2788
                self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy')) # 46.923733
            else:
                self.mean = np.load(pjoin(opt.data_root, 'Mean.npy')) # sum = 49.50974
                self.std = np.load(pjoin(opt.data_root, 'Std.npy')) # sum = 48.370884
                
            print(' mean.sum() = ', self.mean.sum())
            print(' std.sum() = ', self.std.sum())

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        # if args.Mean_evaluator: # 
        #     self.mean = np.load(pjoin(opt.data_root, 'Mean.npy')) # sum = 49.50974
        #     self.std = np.load(pjoin(opt.data_root, 'Std.npy')) # sum = 48.370884
        #     self.mean_for_eval = np.load(pjoin(opt.data_root, 'Mean.npy')) # sum = 49.50974
        #     self.std_for_eval = np.load(pjoin(opt.data_root, 'Std.npy')) # sum = 48.370884

        '''
        用MARDM评估器情况下:
        --gt的mean是67dim的, motion是263的, 在getitem的时候需要motion[:,:dim]-self.mean
        --eval的mean是263dim的会在getitem的时候用到, 提供的motion用不上但shape要求是263用于构造xt
        --mean_for_eval是67dim的, 用于Comp类中的renorm
        '''
        if args.evaluator_eval is not None and 'MARDM' in args.evaluator_eval:
            if mode == 'gt':
                self.mean = np.load(f'/home/deli/project/MARDM/utils/eval_mean_std/{opt.dataset_name}/eval_mean.npy')
                self.std = np.load(f'/home/deli/project/MARDM/utils/eval_mean_std/{opt.dataset_name}/eval_std.npy')
            self.mean_for_eval = np.load(f'/home/deli/project/MARDM/utils/eval_mean_std/{opt.dataset_name}/eval_mean.npy')
            self.std_for_eval = np.load(f'/home/deli/project/MARDM/utils/eval_mean_std/{opt.dataset_name}/eval_std.npy')
        

        self.split_file = pjoin(opt.data_root, f'{split}.txt') # dataset/HumanML3D/train.txt

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = ControlDataset(self.opt, args, self.mean, self.std, self.split_file, self.w_vectorizer, mode,  
                                          control_joint=control_joint, density=density, dataset_name=self.dataset_name, 
                                          normalize_traj=normalize_traj, multi_joint_control=multi_joint_control,
                                          unit_length=unit_length, diffusion=diffusion)
        self.num_actions = 1 # dummy placeholder


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()






def DataLoader(batch_size, args, shuffle=False, mode='train', split='train', num_workers=8, drop_last=True, diffusion=None): 
    
    if batch_size == 1:
        num_workers = 0
        
    if args.no_random == True:
        num_workers = 0
        shuffle = False

    if args.dataset_name == 'snapmogen':
        from dataset.snapmogen_dataset import TextMotionDataset

        # 加载配置
        cfg = load_config('./SnapMoGen/config/eval_momaskplus.yaml')

        # 设置数据路径（数据实际位于 /data/motion/SnapMoGen）
        cfg.data.root_dir = '/data/motion/SnapMoGen'
        cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
        meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
        data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
        all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

        # 根据模式选择数据分割
        if split == 'train':
            mid_split_file = pjoin(data_split_dir, 'train_fnames.txt')
            cid_split_file = pjoin(data_split_dir, 'train_ids.txt')
        else:
            mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
            cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

        # 加载均值和标准差
        # GT 数据始终使用官方 mean/std，因为 evaluator 是在官方归一化数据上训练的
        # snapmogen_no_norm: 不做任何归一化，mean=0, std=1（GT 模式除外）
        if mode == 'gt':
            mean = np.load(pjoin(meta_dir, 'mean.npy'))
            std = np.load(pjoin(meta_dir, 'std.npy'))
            print('[SnapMoGen] GT mode: using official mean/std from', meta_dir)
        elif getattr(args, 'snapmogen_no_norm', False):
            mean = np.zeros(296, dtype=np.float32)
            std = np.ones(296, dtype=np.float32)
            print('[SnapMoGen] snapmogen_no_norm=True: using mean=0, std=1 (no normalization)')
        elif getattr(args, 'correct_snapmogen_norm_all', False):
            norm_dir = './dataset/snapmogen_norm'
            mean = np.load(pjoin(norm_dir, 'mean_all.npy'))
            std = np.load(pjoin(norm_dir, 'std_all.npy'))
            print('[SnapMoGen] Using corrected mean/std (all data) from', norm_dir)
        elif getattr(args, 'correct_snapmogen_norm', False):
            norm_dir = './dataset/snapmogen_norm'
            mean = np.load(pjoin(norm_dir, 'mean.npy'))
            std = np.load(pjoin(norm_dir, 'std.npy'))
            print('[SnapMoGen] Using corrected mean/std from', norm_dir)
        else:
            mean = np.load(pjoin(meta_dir, 'mean.npy'))
            std = np.load(pjoin(meta_dir, 'std.npy'))

        # 使用本地 SnapMoGen 的 TextMotionDataset
        if mode == 'train':
            # 训练模式：启用文本向量化，返回 word_embeddings/pos_one_hots
            #          用 collate_fn 按文本长度降序排序，满足 pack_padded_sequence 要求
            w_vectorizer = WordVectorizer('./glove', 'our_vab')
            opt = get_opt('./dataset/humanml_opt.txt', None)
            dataset = TextMotionDataset(cfg, mean, std, mid_split_file, cid_split_file,
                                        all_caption_path, w_vectorizer=w_vectorizer, opt=opt)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn,
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        elif mode == 'gt':
            # GT 模式：启用文本向量化（供 GRU evaluator 使用 word_embeddings/pos_one_hots）
            #         用 collate_fn 按文本长度降序排序，满足 GRU pack_padded_sequence 要求
            w_vectorizer = WordVectorizer('./glove', 'our_vab')
            opt = get_opt('./dataset/humanml_opt.txt', None)
            dataset = TextMotionDataset(cfg, mean, std, mid_split_file, cid_split_file,
                                        all_caption_path, w_vectorizer=w_vectorizer, opt=opt)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn,
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        else:
            # 评估模式（eval）：基础模式（3 个返回值），向后兼容（gen_loader 训练采样用）
            dataset = TextMotionDataset(cfg, mean, std, mid_split_file, cid_split_file,
                                        all_caption_path)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=None,
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        return train_loader

    if args.dataset_name == 't2m':
        datapath = './dataset/humanml_opt.txt'
        dataset = HumanML3D(mode, args, datapath=datapath, split=split, control_joint=args.control_joint, 
                                    normalize_traj=args.normalize_traj, density=args.density, 
                                    multi_joint_control=args.multi_joint_control, unit_length=2*args.down_t, 
                                    diffusion=diffusion)
    elif args.dataset_name == 'kit':
        datapath = './dataset/kit_opt.txt'
        dataset = HumanML3D(mode, args, datapath=datapath, split=split, control_joint=args.control_joint, 
                                    normalize_traj=args.normalize_traj, density=args.density, 
                                    multi_joint_control=args.multi_joint_control, unit_length=2*args.down_t, 
                                    diffusion=diffusion)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == '__main__':
    train_loader = DataLoader(batch_size=1, mode='train')
    train_loader_iter = cycle(train_loader)