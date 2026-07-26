'''
参考TLControl，一个文本生成多个样本，看每个样本的紧凑程度
'''
import os
import sys
import torch
from datetime import datetime
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OMP_NUM_THREADS'] = '8'
from dataset import dataset_control
from models.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

from data_loaders.humanml.utils.metrics import *
from collections import OrderedDict
from utils.mask_utils import load_ckpt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.fixseed import fixseed
from utils.model_util import create_gaussian_diffusion_simple, get_logger
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset

import seaborn as sns

def plot_tsne(gt_emb, gen_motion, s_point=10):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    N = len(gen_motion)
    gt_emb = gt_emb[:N]
    # 假设 feat_x0: [N, 512], feat_predict: [N, 512]
    all_feats = np.concatenate([gt_emb, gen_motion], axis=0)
    labels = ['GT'] * len(gt_emb) + ['Gen'] * len(gen_motion)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(all_feats)

    plt.scatter(embedding[:,0], embedding[:,1], c=['r' if l=='GT' else 'b' for l in labels], s=s_point)
    plt.title("t-SNE of motion encoder features")
    plt.legend(["GT", "Generated"])
    plt.show()
    
    save_path = os.path.join(os.path.dirname(args.resume_trans), 'tSNE.png')
    plt.savefig(save_path)


def plot_tsne_three_sources(gt_feats, pred_feat1, pred_feat2, title='t-SNE Visualization of Motion Embeddings'):
    """
    Visualize GT, Perceptual-Predicted, and Vanilla-Predicted motion features in 2D using t-SNE.

    Args:
        gt_feats (np.ndarray): Ground-truth motion embeddings, shape (N, D)
        pred_feat1 (np.ndarray): Predicted motion embeddings with perceptual loss, shape (N, D)
        pred_feat2 (np.ndarray): Predicted motion embeddings without perceptual loss, shape (N, D)
        title (str): Plot title
    """
    # 合并特征向量
    N = len(pred_feat1)
    all_feats = np.concatenate([gt_feats[:N], pred_feat1, pred_feat2], axis=0)
    labels = np.array(
        ['GT'] * len(gt_feats) + 
        ['Pred+Perceptual'] * len(pred_feat1) + 
        ['Pred-Vanilla'] * len(pred_feat2)
    )

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feats_2d = tsne.fit_transform(all_feats)

    # 可视化
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=feats_2d[:, 0], 
        y=feats_2d[:, 1], 
        hue=labels,
        palette={'GT': 'green', 'Pred+Perceptual': 'blue', 'Pred-Vanilla': 'red'},
        alpha=0.5,
        s=5
    )
    plt.title(title)
    plt.legend(loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(os.path.dirname(args.resume_trans), 'tSNE3.png')
    plt.savefig(save_path)

def tsne_and_average_distances2(gt_feats, pred_feat1, pred_feat2, title='t-SNE + Distance'):
    """
    使用t-SNE可视化，同时计算GT与预测的点在降维空间的平均距离。
    
    Args:
        gt_feats: [N, D] numpy array, ground truth embeddings
        pred_feat1: [N, D] numpy array, with perceptual loss
        pred_feat2: [N, D] numpy array, without perceptual loss
    """
    N = len(pred_feat1)
    gt_feats = gt_feats[:N]
    assert gt_feats.shape == pred_feat1.shape == pred_feat2.shape

    # 拼接为一个整体进行 t-SNE 降维
    all_feats = np.concatenate([gt_feats, pred_feat1, pred_feat2], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_2d = tsne.fit_transform(all_feats)

    # 拆出降维结果
    gt_2d = all_2d[:N]
    pred1_2d = all_2d[N:2*N]
    pred2_2d = all_2d[2*N:]

    # 计算平均欧氏距离
    dist1 = np.linalg.norm(gt_2d - pred1_2d, axis=1).mean()
    dist2 = np.linalg.norm(gt_2d - pred2_2d, axis=1).mean()

    print(f"[t-SNE] Avg GT vs Pred(+Perceptual) distance: {dist1:.4f}")
    print(f"[t-SNE] Avg GT vs Pred(-Perceptual) distance: {dist2:.4f}")

    # 可视化三类点
    labels = (['GT'] * N + ['Pred+Perceptual'] * N + ['Pred-Vanilla'] * N)
    all_labels = np.array(labels)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=all_2d[:, 0], y=all_2d[:, 1],
        hue=all_labels,
        palette={'GT': 'green', 'Pred+Perceptual': 'blue', 'Pred-Vanilla': 'red'},
        alpha=0.5,
        s=30
    )
    plt.title(title + f'\nAvg Dist(+Perceptual): {dist1:.4f}, Avg Dist(-Perceptual): {dist2:.4f}')
    plt.axis('on')
    plt.legend()
    plt.tight_layout()
    plt.show(); plt.savefig('tSNE3.png')


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    motiontmp = []
    txttmp = []
    emb_dict = {'ground truth':[], 'vald':[], 'vald2':[]}

    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if len(batch) == 7:
                    word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _ = batch
                elif motion_loader_name == 'ground truth':
                    # data_control.py
                    word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, _, _, filename = batch
                else:
                    # comp_v6_model_dataset.py
                    word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, filename = batch
                

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
                
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    gt_emb = torch.cat(emb_dict['ground truth'], axis=0)
    gen_emb = torch.cat(emb_dict['vald'], axis=0)
    gen_emb2 = torch.cat(emb_dict['vald2'], axis=0)
    plot_tsne(gt_emb.cpu().numpy(), gen_emb.cpu().numpy())
    # plot_tsne_three_sources(gt_emb.cpu().numpy(), gen_emb.cpu().numpy(), gen_emb2.cpu().numpy())
    tsne_and_average_distances2(gt_emb.cpu().numpy(), gen_emb.cpu().numpy(), gen_emb2.cpu().numpy())

    return match_score_dict, R_precision_dict, activation_dict

def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
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


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times=1, diversity_times=300, mm_num_times=0, run_mm=False):
    with open(log_file, 'a+') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Control_l2': OrderedDict({}),})

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)


            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

        return


if __name__ == '__main__':

    fixseed(101111)
    args.max_samples = 500
    args.guidance_param = 2.5
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.eval_mode = 'no_mm'

    args.resume_trans = 'output/humanml_enc_512_50steps/model000750000.pt'
    # args.resume_trans = 'output/0605_mdm_el50_50step_find_bestfid/net_best.pth'

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
    log_file += '.log'
    
    if sys.gettrace():
        log_file = f'output/debug/1.log'
    logger = get_logger('', file_path=log_file)
    logger.info(f'*************************************************************')
    # logger.info(f'gtric 文本长度77')
    logger.info(f'log_file = {log_file}')
    logger.info(f'args.dataset_name = {args.dataset_name}')
    logger.info(f'args.modeltype = {args.modeltype}')
    logger.info(f'args.resume_trans = {args.resume_trans}')
    logger.info(f'args.guidance_param = {args.guidance_param}')
    logger.info(f'args.replication_times = {args.replication_times}')
    
    # 主网络
    if args.modeltype in ['diffmdm', 'mdm']:
        from utils.model_util import get_mdm_args
        from models.mdm.model import MDM
        net = MDM(**get_mdm_args(args))
    else:
        raise NotImplementedError

    load_ckpt(net, args.resume_trans, key=None, strict=True)

    if args.guidance_param == 1:
        logger.info('NO CFG !!!!!!!!!!!!!!')
            
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    net.cuda()
    net.eval()

    # 对比的网络
    net2 = MDM(**get_mdm_args(args))
    load_ckpt(net2, 'output/humanml_enc_512_50steps/model000750000.pt', key=None, strict=True)
    diffusion2 = create_gaussian_diffusion_simple(args, net2, args.modeltype)
    net2.cuda()
    net2.eval()
    

    #评估生成数据集部分  shuffle = False
    gt_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='gt', split='test', shuffle=False, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)
    eval_motion_loaders = {
        ## HumanML3D Dataset##
        'vald': lambda: get_control_dataset(
            args, gen_loader, None, None, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit
        ),
        'vald2': lambda: get_control_dataset(
            args, gen_loader, None, None, diffusion2, mm_num_samples, mm_num_repeats, num_samples_limit
        )
    }
    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args)
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
