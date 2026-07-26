import torch
import torch.nn.functional as F

from dataset import dataset_control
import options.option_transformer as option_trans
from utils.model_util import create_gaussian_diffusion_simple
from tqdm import tqdm
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

if __name__ == '__main__':
    args = option_trans.get_args_parser()
    args.batch_size = 32
    args.modeltype = 'MDMBERT'
    args.resume_trans = 'output/0708_MDMBERT_el10_tcl1_infoeval/net_best.pth'


    from models.mdm_bert.mdm_bert import MDMBERT
    from utils.model_util import get_mdm_bert_args
    net = MDMBERT(**get_mdm_bert_args(args, args.modeltype))
    
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode, diffusion=diffusion)
    train_loader_iter = dataset_control.cycle(train_loader)

    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'), args, args.evaluator_eval)

    for nb_iter in tqdm(range(1, args.total_iter+1), position=0, leave=True):
        batch = next(train_loader_iter)
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch



        # text_emb, pred_emb = eval_wrapper.get_co_embeddings_with_grad(word_embeddings, pos_one_hots, sent_len, pred, real_length)
        text_emb, gt_emb = eval_wrapper.get_co_embeddings_with_grad(word_embeddings, pos_one_hots, sent_len, gt_motion, real_length)
        
        # text_cos_loss = 1 - F.cosine_similarity(text_emb, pred_emb, dim=-1).mean()

        text_embeds = F.normalize(text_emb, dim=-1)
        # pred_embeds = F.normalize(pred_emb, dim=-1)
        gt_embeds = F.normalize(gt_emb, dim=-1)
        # pred_similarity = text_embeds @ pred_embeds.T
        gt_similarity = text_embeds @ gt_embeds.T
        a = 1