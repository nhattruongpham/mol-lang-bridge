import numpy as np

import torch
import torch.nn as nn
from transformers import AutoConfig, T5EncoderModel

from .nn import SiLU, linear, timestep_embedding


class TransformerNetModel(nn.Module):
    def __init__(
        self,
        in_channels=32,
        model_channels=128,
        dropout=0.1,
        config_name="QizhiPei/biot5-base-text2mol",
        vocab_size=None,  # 821
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        mask=False,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(config_name)
        config.is_decoder = True
        config.add_cross_attention = True
        config.hidden_dropout_prob = 0.1
        config.num_attention_heads = num_attention_heads
        config.num_hidden_layers = num_hidden_layers
        config.max_position_embeddings = 512
        config.layer_norm_eps = 1e-12

        self.hidden_size = hidden_size
        self.mask = mask
        self.in_channels = in_channels  # 16
        self.model_channels = model_channels  # 128
        self.dropout = dropout
        self.logits_mode = 1
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        self.lm_head = nn.Linear(self.in_channels, vocab_size)
        self.lm_head.weight = self.word_embedding.weight

        self.desc_down_proj = nn.Sequential(
            linear(768, hidden_size),
            SiLU(),
            linear(hidden_size, hidden_size),
        )

        time_embed_dim = model_channels * 4  # 512
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, hidden_size),
        )

        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.input_transformers = T5EncoderModel(config)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, hidden_size
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_down_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, in_channels),
        )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_embeds_with_deep(self, input_ids):
        atom, deep = input_ids
        atom = self.word_embedding(atom)
        deep = self.deep_embedding(deep)

        return torch.concat([atom, deep], dim=-1)

    def get_logits_deep(self, hidden_repr):
        return self.deep_head(hidden_repr)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight**2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(
                text_emb.view(-1, text_emb.size(-1)), 0, 1
            )  # d, bsz*seqlen
            arr_norm = (text_emb**2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = (
                emb_norm
                + arr_norm.transpose(0, 1)
                - 2.0 * torch.mm(self.lm_head.weight, text_emb_t)
            )  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1)
            )  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps, desc_state, desc_mask, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        ################################################################
        if self.mask:
            desc_state = torch.where(timesteps.reshape(-1, 1, 1) < 200, 0.0, desc_state)
            assert len(desc_mask.shape) == 2
            desc_mask = torch.where(timesteps.reshape(-1, 1) < 200, 1.0, desc_mask)
        #################################################################

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        desc_state = self.dropout(self.LayerNorm(self.desc_down_proj(desc_state)))

        input_trans_hidden_states = self.input_transformers(
            emb_inputs,
            encoder_hidden_states=desc_state,
            encoder_attention_mask=desc_mask,
        ).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
