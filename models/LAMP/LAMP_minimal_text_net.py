import torch
import torch.nn as nn
from models.LAMP.QFormer_Base import QFormer_Base

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Qformer, self.query_tokens = QFormer_Base.init_Qformer(
                    num_query_token=49, vision_width=1408, cross_attention_freq=2
                )
        self.tokenizer = QFormer_Base.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self._prepare_qformer()
        self.query_tokens.requires_grad_(False)
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.text2former = nn.Linear(self.Qformer.config.hidden_size, 1408)

    def forward(self, y):
        text_tokens = self.tokenizer(
            y,
            padding="max_length",
            truncation=True,
            max_length=49, # 文本最大长度49
            # max_length=32,
            return_tensors="pt",
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids.cuda(),
            attention_mask=text_tokens.attention_mask.cuda(),
            return_dict=True,
        )

        text_output = self.text2former(text_output.last_hidden_state)
        
        cond_vector = torch.mean(text_output, dim=1)


        # 下面这一段都是text-grounded motion generation的训练，严格来说不能算文本
        # text_atts = torch.ones(text_output.size()[:-1], dtype=torch.long).to(             # [bs, 49]
        #         text_output.device
        # )
        # query_tokens = self.query_tokens.expand(text_output.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        # text_output = self.Qformer.bert(
        #     query_embeds=query_tokens,              # [bs, 49, 768]
        #     encoder_hidden_states=text_output,     # [bs, 49, 1408]
        #     encoder_attention_mask=text_atts,      # [bs, 49]
        #     use_cache=True,
        #     return_dict=True,
        # )
        # cond_vector = torch.mean(text_output.last_hidden_state, dim=1)

        return cond_vector

    def _prepare_qformer(self):
        ckpt = torch.load('/home/deli/project/LaMP/checkpoints/t2m/h3d-qformer.tar', map_location='cpu')
        base_ckpt = {k.replace("Qformer.", ""): v for k,
                    v in ckpt['motion_qformer'].items()}
        unexpected_keys, missing_keys = self.Qformer.load_state_dict(base_ckpt, strict=False)
        self.query_tokens = nn.Parameter(base_ckpt['query_tokens'])


if __name__ == '__main__':
    net = Net()
    text = ('a person is running',)
    emb = net(text)
