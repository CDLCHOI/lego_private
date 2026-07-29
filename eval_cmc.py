
import sys
# sys.path.append('.')
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OMP_NUM_THREADS'] = '8'
from utils.fixseed import fixseed
import torch
import torch.nn.functional as F
from data_loaders.humanml.utils.metrics import *
from datetime import datetime
import numpy as np
from collections import OrderedDict
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
from dataset import dataset_control
import ipdb
import warnings
warnings.filterwarnings('ignore')
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper, EvaluatorMARDM
from data_loaders.humanml.networks.evlauator_wrapper_salad import EvaluatorModelWrapperSALAD
from utils.mask_utils import load_ckpt
from utils.motion_process import recover_from_ric, recover_from_rot, recover_root_rot_pos

from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import cont6d_to_quat

from utils.mask_utils import vis_motion




def evaluate_control(motion_loaders, file):
    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    print('========== Evaluating Control ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    traj_err = []
    traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    # print(motion_loader_name)
    
    # 初始化骨骼
    example_data = np.load('/data/motion/HumanML3D/new_joints/000021.npy')
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    skel.get_offsets_joints(example_data[0])
    
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader): # 这里就是Comp类出来的已经用mean_for_eval归一化过的motion
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, hint, filename = batch
            # process motion
            # sample to motion
            dim = motions.shape[-1]
            mean_for_eval = motion_loader.dataset.gen_loader.dataset.mean_for_eval[:dim]
            std_for_eval = motion_loader.dataset.gen_loader.dataset.std_for_eval[:dim]
            motions = motions * std_for_eval + mean_for_eval
            # vis_motion(motion1=sample[0], motion2=gt_motion[0], save_path='visualization/1.html', vis=True)
            motions = motions.float()
            n_joints = 22 if motions.shape[-1] in [263, 67] else 21
            
            # joints = recover_from_ric(motions, n_joints)
            joints = recover_from_ric(motions, n_joints)

            if n_joints == 21:
                # kit
                joints = joints * 0.001
            

            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(joints.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(joints.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # control l2 error
            # process hint
            mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(dim=-1, keepdim=True) != 0
            raw_mean = motion_loader.dataset.gen_loader.dataset.t2m_dataset.raw_mean
            raw_std = motion_loader.dataset.gen_loader.dataset.t2m_dataset.raw_std
            hint = hint * raw_std + raw_mean
            if n_joints == 21:
                hint = hint * 0.001
            hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask_hint
            i = 0
            for motion, h, mask in zip(joints, hint, mask_hint):
                control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                mean_error = control_error.sum() / mask.sum()
                dist_sum += mean_error
                control_error = control_error.reshape(-1)
                mask = mask.reshape(-1)
                err_np = calculate_trajectory_error(control_error, mean_error, mask)
                traj_err.append(err_np)
                # ferr.write(f'{filename[i]} {mean_error.item():.4f} {control_error.max():.4f}\n')
                i += 1

            all_size += joints.shape[0]

        # l2 dist
        dist_mean = dist_sum / all_size
        l2_dict[motion_loader_name] = dist_mean

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err

    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
    line = f'---> [{motion_loader_name}] Trajectory Error: '
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))
    print(line)
    print(line, file=file, flush=True)
    return l2_dict, skating_ratio_dict, trajectory_score_dict

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    match_score_sim_dict = OrderedDict({})
    R_precision_sim_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    clip_score_dict = OrderedDict({}) # 加入CLIPscore指标

    print('========== Evaluating Matching Score ==========')
    motiontmp = []
    txttmp = []
    emb_dict = {'ground truth':[], 'vald':[]}

    for motion_loader_name, motion_loader in motion_loaders.items():
        # TMR evaluator: GT 数据在 meta 域, vald 数据在 hml 域
        if hasattr(eval_wrapper, 'set_norm_domain'):
            domain = 'meta' if motion_loader_name == 'ground truth' else 'hml'
            eval_wrapper.set_norm_domain(domain)

        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0

        topk_count_sim_sum = 0
        matching_score_sim_sum = 0

        clip_score_real = 0

        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if args.dataset_name == 'snapmogen':
                    # SnapMoGen T5 evaluator
                    caption, motions, m_lens = batch
                    motions = motions.float().cuda()[...,:148]
                    text_embeddings, _ = eval_wrapper.encode_text(caption, sample_mean=True)
                    _, motion_embeddings, _ = eval_wrapper.encode_motion(motions, m_lens, sample_mean=True)
                else:
                    if len(batch) == 7:
                        word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _ = batch
                    elif motion_loader_name == 'ground truth':
                        # data_control.py
                        word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, _, _, filename = batch
                    else:
                        # comp_v6_model_dataset.py
                        word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, filename = batch

                    if args.evaluator_eval is not None and 'MARDM' in args.evaluator_eval:
                        (text_embeddings, motion_embeddings), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
                            word_embs=word_embeddings,
                            pos_ohot=pos_one_hots,
                            cap_lens=sent_lens,
                            captions=caption,
                            motions=motions,
                            m_lens=m_lens
                        )
                    else:
                        if args.evaluator_eval_type == 'tmr':
                            # TMR evaluator 使用 raw text (captions) 编码文本
                            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                                word_embs=word_embeddings,
                                pos_ohot=pos_one_hots,
                                cap_lens=sent_lens,
                                motions=motions,
                                m_lens=m_lens,
                                captions=caption
                            )
                        else:
                            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                                word_embs=word_embeddings,
                                pos_ohot=pos_one_hots,
                                cap_lens=sent_lens,
                                motions=motions,
                                m_lens=m_lens
                            )
                # except: 67的评估器，有需要在用
                #     text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, caption, motions[...,:67], m_lens)

                emb_dict[motion_loader_name].append(motion_embeddings)
                
                # 老指标，R precision distance
                # print('text_embeddings.sum()=', text_embeddings.sum())
                # print('motion_embeddings.sum()=', motion_embeddings.sum())
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1) # (32,32)  沿着1轴对每个元素进行虚空排序，并返回排序后的元素在原矩阵中的索引；所以如果是一个非常正确的匹配，argsmax[0,0]=0,argsmax[1,0]=1,argsmax[2,0]=2 ......；argsmax的第i行从左到右都是索引，argsmax[i,0]表示最匹配的数值在原矩阵的索引，argsmax[i,1]表示次匹配的数值在原矩阵的索引
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0) # (3,)  就是一个32的batch中，top1 top2 top3的值


                # 新指标，cosine similarity
                text_embeds = F.normalize(text_embeddings, dim=-1)
                motion_embeds = F.normalize(motion_embeddings, dim=-1)
                sim_mat = text_embeds @ motion_embeds.T
                matching_score_sim = sim_mat.trace()
                # if ipdb.set_trace():
                #     a = 1
                argsmax_sim = np.argsort((0-sim_mat).cpu().numpy(), axis=1)
                top_k_sim_mat = calculate_top_k(argsmax_sim, top_k=3)
                top_k_sim_count = top_k_sim_mat.sum(axis=0) 
                topk_count_sim_sum += top_k_sim_count
                matching_score_sim_sum += matching_score_sim.cpu().numpy()


                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

                if args.evaluator_eval is not None and 'MARDM' in args.evaluator_eval:
                # 加个MARDM的CLIPscore
                    batch_clip_score_pred = 0
                    for j in range(32):
                        single_em = em_pred_clip[j]
                        single_et = et_pred_clip[j]
                        clip_score = (single_em @ single_et.T).item()
                        batch_clip_score_pred += clip_score
                    clip_score_real += batch_clip_score_pred

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size # 然后这里取个平均
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings
            # 加入sim相似度指标
            matching_score_sim = matching_score_sim_sum / all_size
            R_precision_sim = topk_count_sim_sum / all_size
            match_score_sim_dict[motion_loader_name] = matching_score_sim
            R_precision_sim_dict[motion_loader_name] = R_precision_sim

            if args.evaluator_eval is not None and 'MARDM' in args.evaluator_eval:
                # 加入CLIPscore指标
                clip_score_real /= all_size
                clip_score_dict[motion_loader_name] = clip_score_real


        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

        # 打印相似度指标
        print(f'---> [{motion_loader_name}] Matching Similarity: {matching_score_sim:.4f}')
        print(f'---> [{motion_loader_name}] Matching Similarity: {matching_score_sim:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision_sim: '
        for i in range(len(R_precision_sim)):
            line += '(top %d): %.4f ' % (i+1, R_precision_sim[i])
        print(line)
        print(line, file=file, flush=True)

        # CLIP score
        print(f'---> [{motion_loader_name}] CLIP Score: {clip_score_real:.4f}')
        print(f'---> [{motion_loader_name}] CLIP Score: {clip_score_real:.4f}', file=file, flush=True)

    gt_emb = torch.cat(emb_dict['ground truth'], axis=0)
    gen_emb = torch.cat(emb_dict['vald'], axis=0)
    # plot_tsne(gt_emb.numpy(), gen_emb.numpy())

    return match_score_dict, R_precision_dict, activation_dict, match_score_sim_dict, R_precision_sim_dict, clip_score_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')

    # TMR evaluator: GT 数据在 meta 归一化域
    if hasattr(eval_wrapper, 'set_norm_domain'):
        eval_wrapper.set_norm_domain('meta')

    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            if args.dataset_name == 'snapmogen':
                caption, motions, m_lens = batch
                motions = motions.float().cuda()[...,:148]
                _, motion_embeddings, _ = eval_wrapper.encode_motion(motions, m_lens, sample_mean=False)
            else:
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _, filename= batch
                if eval_wrapper.dim_pose < 100:
                    motions=motions[...,:67]

                motion_embeddings = eval_wrapper.get_motion_embeddings(
                    motions=motions,
                    m_lens=m_lens
                )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')

    # TMR evaluator: 生成数据在 hml 归一化域
    if hasattr(eval_wrapper, 'set_norm_domain'):
        eval_wrapper.set_norm_domain('hml')

    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times=1, diversity_times=300, mm_num_times=0, run_mm=False):
    with open(log_file, 'a+') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'Matching Score Sim': OrderedDict({}),
                                   'R_precision_sim': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Control_l2': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'Trajectory Error': OrderedDict({}),
                                   'CLIP Score': OrderedDict({})},)

        # 用于可视化的 motion 数据（仅 SnapMoGen 有效）
        vis_motion_data = None

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            # 取 vald loader 的第一个 batch 的第 0 个 motion 用于可视化
            if args.dataset_name == 'snapmogen' and vis_motion_data is None:
                vald_loader = motion_loaders.get('vald')
                if vald_loader is not None:
                    try:
                        first_batch = next(iter(vald_loader))
                        caption, motions, m_lens = first_batch
                        # 获取数据集 mean/std 用于后续反归一化
                        dataset_obj = vald_loader.dataset
                        ds_mean = dataset_obj.dataset.mean
                        ds_std = dataset_obj.dataset.std
                        vis_motion_data = {
                            'motion': motions[0].copy(),       # (L, 296)
                            'm_length': int(m_lens[0]),
                            'caption': str(caption[0]),
                            'mean': ds_mean.copy(),
                            'std': ds_std.copy(),
                        }
                    except Exception as e:
                        print(f'[WARNING] 获取可视化 motion 失败: {e}')

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            if args.dataset_name != 'snapmogen':
                control_l2_dict, skating_ratio_dict, trajectory_score_dict = evaluate_control(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict, mat_score_sim_dict, R_precision_sim_dict, clip_score_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)
            

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            if args.dataset_name != 'snapmogen':
                for key, item in skating_ratio_dict.items():
                    if key not in all_metrics['Skating Ratio']:
                        all_metrics['Skating Ratio'][key] = [item]
                    else:
                        all_metrics['Skating Ratio'][key] += [item]

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in mat_score_sim_dict.items():
                if key not in all_metrics['Matching Score Sim']:
                    all_metrics['Matching Score Sim'][key] = [item]
                else:
                    all_metrics['Matching Score Sim'][key] += [item]

            for key, item in R_precision_sim_dict.items():
                if key not in all_metrics['R_precision_sim']:
                    all_metrics['R_precision_sim'][key] = [item]
                else:
                    all_metrics['R_precision_sim'][key] += [item]

            for key, item in clip_score_dict.items():
                if key not in all_metrics['CLIP Score']:
                    all_metrics['CLIP Score'][key] = [item]
                else:
                    all_metrics['CLIP Score'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        # ipdb.set_trace()
        for metric_name, metric_dict in all_metrics.items(): # odict_keys(['Matching Score', 'R_precision', 'FID', 'Diversity', 'MultiModality', 'Control_l2', 'Skating Ratio', 'Trajectory Error'])
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif metric_name == 'Trajectory Error':
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)): # zip(traj_err_key, mean):
                        line += '(%s): Mean: %.4f CInt: %.4f; ' % (traj_err_key[i], mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict, vis_motion_data



if __name__ == '__main__':
    from utils.model_util import create_gaussian_diffusion_simple, get_logger
    # fixseed(args.seed)
    args.guidance_param = 2.5
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    

    assert args.resume_trans is not None, 'Must specify resume_trans'
    
    # 计算multimodality：batchsize=32，test split有145次迭代，里面挑选3次迭代，每次迭代里同一个文本都跑30次生成，即得到了(96，30, dim)的动作特征。从(96,30,dim)里选取两次(96,10,dim)，计算距离，即为multimodality
    args.normalize_traj = True # 归一化轨迹再输入
    if args.eval_mode == 'no_mm':
        num_samples_limit = args.max_samples
        run_mm = False
        mm_num_samples = 0 #  100
        mm_num_repeats = 0 # 一个文本生成几次动作, 30次
        mm_num_times = 0 # 10   算multimodality
        diversity_times = 300
        replication_times = args.replication_times # 重复测试次数
    elif args.eval_mode == 'with_mm':
        num_samples_limit = args.max_samples
        run_mm = True
        mm_num_samples = 100 #  100
        mm_num_repeats = 30 # 一个文本生成几次动作, 30次
        mm_num_times = 10 # 10   算multimodality
        diversity_times = 300
        replication_times = args.replication_times # 重复测试次数
    else:
        raise ValueError()


    log_file = f"{os.path.dirname(args.resume_trans)}/t2m"
    log_file += f'_repeat{args.replication_times}'
    if args.guidance_param != 2.5:
        log_file += f'_scale{args.guidance_param}'
    log_file += f'_num{args.max_samples}'
    if 'best' in args.resume_trans:
        log_file += '_net_best'
    elif 'last' in args.resume_trans:
        log_file += '_net_last'
    elif 'best_diff' in args.resume_trans:
        log_file += '_best_diff'
    log_file += '.log'
    
    if sys.gettrace():
        log_file = f'output/debug/1.log'
    logger = get_logger('', file_path=log_file)
    
    logger.info(f'*************************************************************')
    
    cmd = "python " + " ".join(sys.argv)
    logger.info(cmd)
    logger.info(f'log_file = {log_file}')
    logger.info(f'args.dataset_name = {args.dataset_name}')
    logger.info(f'args.modeltype = {args.modeltype}')
    logger.info(f'args.resume_trans = {args.resume_trans}')
    logger.info(f'args.guidance_param = {args.guidance_param}')
    logger.info(f'args.replication_times = {args.replication_times}')
    logger.info(f'args.evaluator_eval = {args.evaluator_eval}')
    
    

    if args.modeltype in ['diffmdm', 'mdm']:
        from utils.model_util import get_mdm_args
        from models.mdm.model import MDM
        net = MDM(**get_mdm_args(args))
    elif args.modeltype in ['mdm_bert']:
        from models.mdm_bert.mdm_bert import MDMBERT
        from utils.model_util import get_mdm_bert_args
        net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))
    else:
        raise NotImplementedError

    load_ckpt(net, args.resume_trans, key=None, strict=False)

    if args.guidance_param == 1:
        logger.info('NO CFG !!!!!!!!!!!!!!')
            
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    net.cuda()
    net.eval()
    

    #评估生成数据集部分  shuffle = False
    gt_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='gt', split='test', shuffle=True, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    eval_motion_loaders = {
        ## HumanML3D Dataset##
        'vald': lambda: get_control_dataset(
            args, gen_loader, None, None, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit
        )
    }

    if args.evaluator_eval is not None:
        print(f'=== using evaluator_eval {args.evaluator_eval}')
        if 'salad' in args.evaluator_eval:
            opt = args
            opt.device = 'cuda:0' # 自己加的
            opt.latent_dim = 32
            opt.activation = 'gelu'
            opt.n_layers = 2
            opt.n_extra_layers = 1
            opt.kernel_size = 3 
            opt.norm = 'none'
            opt.dropout = 0.1
            eval_wrapper = EvaluatorModelWrapperSALAD(opt, args.evaluator_eval)
        elif 'MARDM' in args.evaluator_eval:
            eval_wrapper = EvaluatorMARDM(args.dataset_name, torch.device('cuda'))
        else:
            eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args, args.evaluator_eval)
    else:
        eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args, args.evaluator_eval)
    # eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args, args.evaluator_eval)


    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
    



