import torch
import options.option_transformer as option_trans
import os 
import numpy as np
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt, vis_motion
from utils.model_util import create_gaussian_diffusion_simple
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
import matplotlib.colors as mcolors
import textwrap
from data_loaders.humanml.utils.metrics import euclidean_distance_matrix
from utils.motion_process import recover_from_ric, recover_from_rot, t2m_tgt_skel, t2m_tgt_offsets, t2m_raw_offsets

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

    load_ckpt(net, ckpt_path, key=None, strict=False)
    diffusion = create_gaussian_diffusion_simple(args, net, modeltype)
    net.cuda()
    net.eval()

    return net, diffusion

def wrap_label(label, width=70):
    return '\n'.join(textwrap.wrap(label, width))


def visualize_two_models_tsne_by_shade(
    pred_motion_embeddings_1, 
    pred_motion_embeddings_2, 
    text_embeddings,
    texts, 
    samples_per_text=10, 
    perplexity=10,
    model_names=("Model 1", "Model 2", 'Text')
):
    """
    用颜色深浅表示不同模型，同一文本使用同一色调。
    """
    # 转 numpy
    if isinstance(pred_motion_embeddings_1, torch.Tensor):
        pred_motion_embeddings_1 = pred_motion_embeddings_1.cpu().numpy()
    if isinstance(pred_motion_embeddings_2, torch.Tensor):
        pred_motion_embeddings_2 = pred_motion_embeddings_2.cpu().numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.cpu().numpy()

    num_texts = len(texts)
    total_samples = num_texts * samples_per_text

    assert pred_motion_embeddings_1.shape[0] == total_samples
    assert pred_motion_embeddings_2.shape[0] == total_samples
    assert text_embeddings.shape[0] == num_texts

    # 合并 embeddings
    # all_embeddings = np.vstack([pred_motion_embeddings_1, pred_motion_embeddings_2])
    all_embeddings = np.vstack([
        pred_motion_embeddings_1,
        pred_motion_embeddings_2,
        text_embeddings
    ])

    # 标签构建：[text_idx, model_idx]
    label_pairs = []
    for text_idx in range(num_texts):
        label_pairs += [(text_idx, 0)] * samples_per_text  # model 1
    for text_idx in range(num_texts):
        label_pairs += [(text_idx, 1)] * samples_per_text  # model 2
    for text_idx in range(num_texts):
        label_pairs += [(text_idx, 2)]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # 设置颜色：为每个文本分配主色
    base_colors = sns.color_palette("hls", num_texts)  # hls 色调区分文本

    # 构建颜色字典（颜色深浅区分模型）
    color_dict = {}
    for text_idx, base_color in enumerate(base_colors):
        deep = mcolors.to_rgba(base_color, alpha=0.2)    # 模型1 - 浅色
        light = mcolors.to_rgba(base_color, alpha=0.8)   # 模型2 - 深色
        solid = mcolors.to_rgba(base_color, alpha=1.0)   # 文本embedding
        color_dict[(text_idx, 0)] = deep
        color_dict[(text_idx, 1)] = light
        color_dict[(text_idx, 2)] = solid

    # 绘图
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid", font_scale=1.1)

    for text_idx in range(num_texts):
        for model_idx in [0, 1, 2]:
            indices = [i for i, (t, m) in enumerate(label_pairs) if t == text_idx and m == model_idx]
            plt.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                label=wrap_label(f"{texts[text_idx]}") if model_names[model_idx] == 'Text' else None,
                color=color_dict[(text_idx, model_idx)],
                s=180 if model_names[model_idx] == 'Text' else 40,
                edgecolors='none'
            )

    # plt.title("t-SNE of Motion Embeddings: Shade = Model, Hue = Text")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")

    # 图例放右边
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=True)
    # plt.subplots_adjust(left=0.2)
    # plt.subplots_adjust(right=0.2)
    plt.subplots_adjust(bottom=0.5)  #  给图下方留出 25% 空间
    plt.gca().set_aspect('equal', adjustable='datalim') # 坐标区域保持接近正方形
    plt.savefig(f'tsne_{num_texts}texts_{args.batch_size}samples.pdf')
    plt.show()

if __name__ == '__main__':
    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args)
    # fixseed(1234)
    args.dataset_name = 't2m'
    args.batch_size = 1
    num_texts = 3

    # 对照组，50步，干净数据训练，有无el
    # args.diffusion_steps = 50
    # args.resume_trans1 = 'output/humanml_enc_512_50steps/model000750000.pt'; args.modeltype1 = 'mdm'
    # # args.resume_trans2 = 'output/0605_mdm_el50_50step_find_bestfid/net_best.pth'; args.modeltype2 = 'mdm'
    # args.resume_trans2 = 'output/0605_mdm_step50/net_best.pth'; args.modeltype2 = 'mdm'
    # args.resume_trans2 = 'output/0617_mdm_50step_el5/net_best.pth'; args.modeltype2 = 'mdm'
    # # args.resume_trans2 = 'output/0612_MDMBERT_el50_tel50/net_best.pth'; args.modeltype2 = 'mdm_bert'
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_clean_w_wo_el/'

    # 对照组  0-50随机加噪
    # args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'
    # args.resume_trans = 'output/0528_mdm_step50_noisy50/net_last.pth'; args.diffusion_steps = 50
    # args.resume_trans2 = 'output/0528_mdm_el50_step50_noisy50/net_last.pth'; args.diffusion_steps = 50
    # args.resume_trans3 = None
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_random0_50_noisy_w_wo_el/'

    # 对照组 固定10步加噪训练
    # args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'; args.diffusion_steps = 50
    # args.resume_trans = 'output/0531_mdm_step50_noisy1010/net_last.pth'; 
    # args.resume_trans2 = 'output/0531_mdm_step50_noisy1010_el50/net_last.pth'
    # args.resume_trans3 = 'output/0601_mdm_step50_noisy1010_el50_union/net_last.pth'
    # dir_path = f'visualization/compare_{args.diffusion_steps}step_w_wo_el/'

    # el与tel的高R precision对比
    # args.modeltype1 = 'mdm_bert'; args.modeltype2 = 'mdm_bert'; args.diffusion_steps = 50
    # # args.resume_trans1 = 'output/humanml_trans_dec_512_bert/model000600000.pt'
    # args.resume_trans1 = 'output/0609_MDMBERT/net_best.pth'
    # args.resume_trans2 = 'output/0612_MDMBERT_el50_tel50/net_best.pth'

    # el与tel的高R precision对比  MDM
    args.modeltype1 = 'mdm'; args.modeltype2 = 'mdm'; args.diffusion_steps = 50
    args.resume_trans1 = 'output/humanml_enc_512_50steps/model000750000.pt'
    # args.resume_trans2 = 'output/0616_mdm_el50_tel10/net_best.pth'
    # args.resume_trans2 = 'output/0617_mdm_50step_el5/net_best.pth'
    args.resume_trans2 = 'output/0625_mdm_50step_tel20_ricrot/net_best.pth'
    
    # args.resume_trans2 = 'output/0616_mdm_el50_tel20/net_best.pth'


    
    net1, diffusion1 = build_model(args, args.modeltype1, args.resume_trans1)
    net2, diffusion2 = build_model(args, args.modeltype2, args.resume_trans2)


    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    
    text_length_list = []
    text_list = []
    emb1_list = []
    emb2_list = []
    text_emb_list = []

    for i, batch in enumerate(gen_loader):
        word_embeddings, pos_one_hots, clip_text, sent_lens, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)

        copyed_real_length = real_length[0].repeat(args.batch_size)
        copyed_text = clip_text[0:1] * args.batch_size
        copyed_word_embeddings = word_embeddings[0].repeat(args.batch_size,1,1)
        copyed_pos_one_hots = pos_one_hots[0].repeat(args.batch_size,1,1)
        copyed_sent_len = sent_lens[0].repeat(args.batch_size)

        model_kwargs = {}
        model_kwargs['real_mask'] = real_mask
        model_kwargs['clip_text'] = copyed_text
        print('copyed_text = ', copyed_text[0])
        text_list.append(copyed_text[0])
        
        x0 = gt_motion

        sample1 = diffusion1.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=args.batch_size)
        sample2 = diffusion2.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=args.batch_size)
        
        gt_emb = eval_wrapper.get_motion_embeddings_with_grad(motions=x0[0:1], m_lens=copyed_real_length[0:1])
        emb1 = eval_wrapper.get_motion_embeddings_with_grad(motions=sample1, m_lens=copyed_real_length)
        emb2 = eval_wrapper.get_motion_embeddings_with_grad(motions=sample2, m_lens=copyed_real_length)
        text_emb = eval_wrapper.get_co_embeddings(copyed_word_embeddings, copyed_pos_one_hots, copyed_sent_len, sample1, copyed_real_length)[0]

        emb1_list.append(emb1.detach().cpu().numpy())
        emb2_list.append(emb2.detach().cpu().numpy())
        # text_emb_list.append(gt_emb.detach().cpu().numpy()[0:1]) # 存gt_motion的embedding
        text_emb_list.append(text_emb.detach().cpu().numpy()[0:1]) # 取第一个，因为后面都是重复的
        text_length_list.append(copyed_sent_len[0])


        ########################################################################
        # 这部分是为了查看同个文本，2个模型各进行10batch生成，看看matching score是否真的能提现出文本动作匹配度 
        unnormed_sample2 = gen_loader.dataset.t2m_dataset.inv_transform(sample2.cpu().numpy())
        unnormed_sample2 = torch.from_numpy(unnormed_sample2).cuda()
        # bone_loss = diffusion1._calc_bone_loss(sample2)

        vis_motion(motion1=sample1[0], motion2=sample1[0], vis=True, using_rotation=False, joint2_from_joint1rot=True)
        ric_rot_loss1, dist1 = diffusion1._calc_ric_rot_loss(sample1[0:1], real_mask[0:1], return_dist=True, need_denorm=True)
        print(f'ric_rot_loss1 = {ric_rot_loss1.item()}, dist1 = {dist1.item()}')

        
        vis_motion(motion1=sample2[0], motion2=sample2[0], vis=True, using_rotation=False, joint2_from_joint1rot=True)
        ric_rot_loss2, dist2 = diffusion1._calc_ric_rot_loss(sample2[0:1], real_mask[0:1], return_dist=True, need_denorm=True)
        print(f'ric_rot_loss2 = {ric_rot_loss2.item()}, dist2 = {dist2.item()}')

        save_dir = 'visualization/matching_score'
        with open(os.path.join(save_dir, f'text{i}_mlen_{copyed_real_length[0]}.txt'),'a') as f:
            f.write(str(copyed_text[0]))
        dist_mat1 = euclidean_distance_matrix(text_emb.detach().cpu().numpy(), emb1.detach().cpu().numpy())
        dist_mat2 = euclidean_distance_matrix(text_emb.detach().cpu().numpy(), emb2.detach().cpu().numpy())
        argsmax = np.argsort(dist_mat1, axis=1)
        matching_score1 = np.diag(dist_mat1)
        matching_score2 = np.diag(dist_mat2)
        print(f'Text{i} \nmatching_score1 = {matching_score1} \nmatching_score2 = {matching_score2}, real_length = {copyed_real_length[0]}')
        for j in range(args.batch_size):
            save_name = os.path.join(save_dir, f'text{i}_b{j}_{matching_score1[j]:.2f}_{matching_score2[j]:.2f}.html')
            vis_motion(motion1=sample1[j], motion2=sample2[j], save_path=save_name, vis=True)
            save_name = os.path.join(save_dir, f'text{i}_b{j}_{matching_score1[j]:.2f}_{matching_score2[j]:.2f}_rot.html')
            vis_motion(motion1=sample2[j], motion2=sample2[j], save_path=save_name, vis=True, using_rotation=False, joint2_from_joint1rot=True)
            a = 1
        a = 1
        ##############################################################################

        
        if i==num_texts-1:
            break

    # 这部分是画tSNE的，可以注释掉
    # emb1 = np.concatenate(emb1_list, axis=0)
    # emb2 = np.concatenate(emb2_list, axis=0)
    # text_emb = np.concatenate(text_emb_list, axis=0)
    # print('text_length_list= ', text_length_list)
    # visualize_two_models_tsne_by_shade(emb1, emb2, text_emb, text_list, samples_per_text=args.batch_size)

    

    
    
    
    