# import os
import torch
import torch.nn as nn
# from typing import Optional, Union, Tuple
# from dataclasses import dataclass

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from transformers import AutoTokenizer

# from argparse import ArgumentParser


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


# @dataclass
# class CausalLMOutput:
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = None
        self.args = args
        self.tokenizer_dir = 'meta-llama/Llama-2-7b-hf' if args.hf_dir is None else args.hf_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

        self.config = DecoderConfig()
        # self.config.override(vars(self.args))
        self.config.__dict__.update(vars(self.args))
        self.config.vocab_size = len(self.tokenizer)

        self.build_model()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def build_model(self):
        if self.args.model_name_trm == "trm_medium":
            self.config.decoder_embed_dim = 1024
            self.config.decoder_layers = 24
            self.config.decoder_attention_heads = 16
        elif self.args.model_name_trm == "trm_large":
            self.config.decoder_embed_dim = 1536
            self.config.decoder_layers = 24
            self.config.decoder_attention_heads = 16
        elif self.args.model_name_trm == "trm_3b":
            self.config.decoder_embed_dim = 2560
            self.config.decoder_layers = 32
            self.config.decoder_attention_heads = 32
        elif self.args.model_name_trm == "trm_7b":
            self.config.decoder_embed_dim = 4096
            self.config.decoder_layers = 32
            self.config.decoder_attention_heads = 32
        elif self.args.model_name_trm == "trm_13b":
            self.config.decoder_embed_dim = 5120
            self.config.decoder_layers = 40
            self.config.decoder_attention_heads = 40
        elif self.args.model_name_trm == "trm_65b":
            self.config.decoder_embed_dim = 8192
            self.config.decoder_layers = 64
            self.config.decoder_attention_heads = 64
        self.config.decoder_ffn_embed_dim = self.config.decoder_embed_dim * 4
        self.config.n_layer = self.config.decoder_layers
        self.config.n_embd = self.config.decoder_embed_dim

        embed_tokens = Embedding(self.config.vocab_size, self.config.decoder_embed_dim)
        self.model = Decoder(self.config, embed_tokens=embed_tokens)

    def max_positions(self):
        return self.config.max_target_positions

    def forward(self, input_ids: torch.LongTensor = None):
        logits, aux_dic = self.model(input_ids)
        return logits
