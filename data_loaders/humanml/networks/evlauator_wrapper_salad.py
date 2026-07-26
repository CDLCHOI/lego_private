from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.salad_encdec import MotionEncoderSALAD
from utils.word_vectorizer import POS_enumerator


class EvaluatorModelWrapperSALAD(object):

    def __init__(self, opt, ckpt_path):
        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
            opt.contact_joints = [7, 10, 8, 11]
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
            [19, 20, 14, 15]
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        opt.dim_movement_latent = 512
        opt.unit_length = 4

        self.opt = opt
        self.dim_pose = opt.dim_pose # 勿删 eval_cmc.py里用到 eval_wrapper.dim_pose < 100

        self.text_encoder, self.motion_encoder, self.movement_encoder = self.build_models(ckpt_path)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.train()
        self.motion_encoder.train()
        self.movement_encoder.train()

    def build_models(self, ckpt_path):
        opt = self.opt
        movement_enc = MotionEncoderSALAD(opt)
        text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                    pos_size=opt.dim_pos_ohot,
                                    hidden_size=opt.dim_text_hidden,
                                    output_size=opt.dim_coemb_hidden,
                                    device=opt.device)

        motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                        hidden_size=opt.dim_motion_hidden,
                                        output_size=opt.dim_coemb_hidden,
                                        device=opt.device)
        assert ckpt_path is not None, 'Please provide a valid ckpt_path'
        
        checkpoint = torch.load(ckpt_path, map_location=opt.device)
        movement_enc.load_state_dict(checkpoint['movement_encoder'])
        text_enc.load_state_dict(checkpoint['text_encoder'])
        motion_enc.load_state_dict(checkpoint['motion_encoder'])
        print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
        return text_enc, motion_enc, movement_enc

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
    
    def get_motion_embeddings_with_grad(self, motions, m_lens):
        with torch.enable_grad():
            # motions = motions.detach().to(self.device).float()
            motions = motions.to(self.device).float() # 去掉detach

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            # movements = self.movement_encoder(motions[..., :-4]).detach()
            movements = self.movement_encoder(motions) # 去掉detach  把[..., :-4]去掉了
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding