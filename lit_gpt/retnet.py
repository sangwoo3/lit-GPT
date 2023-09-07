import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from dataclasses import dataclass

from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder
from transformers import AutoTokenizer

from lit_gpt.utils import chunked_cross_entropy


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


# @dataclass
# class CausalLMOutput:
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None


class RetNet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = None
        self.args = args
        self.model_name_tokenizer = 'meta-llama/Llama-2-7b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_tokenizer)

        self.config = RetNetConfig()
        # self.config.override(self.args)
        self.config.vocab_size = len(self.tokenizer)

        self.build_model()

    def build_model(self):
        if self.args.model_name == "retnet_medium":
            self.config.decoder_embed_dim = 1024
            self.config.decoder_ffn_embed_dim = 2048
            self.config.decoder_layers = 16
            self.config.decoder_retention_heads = 4
        elif self.args.model_name == "retnet_xl":
            self.config.decoder_embed_dim = 2048
            self.config.decoder_ffn_embed_dim = 4096
            self.config.decoder_layers = 24
            self.config.decoder_retention_heads = 8
        elif self.args.model_name == "retnet_3b":
            self.config.decoder_embed_dim = 2560
            self.config.decoder_ffn_embed_dim = 5120
            self.config.decoder_layers = 32
            self.config.decoder_retention_heads = 10
        elif self.args.model_name == "retnet_7b":
            self.config.decoder_embed_dim = 4096
            self.config.decoder_ffn_embed_dim = 8192
            self.config.decoder_layers = 32
            self.config.decoder_retention_heads = 16
        elif self.args.model_name == "retnet_13b":
            self.config.decoder_embed_dim = 5120
            self.config.decoder_ffn_embed_dim = 10240
            self.config.decoder_layers = 40
            self.config.decoder_retention_heads = 20
        elif self.args.model_name == "retnet_65b":
            self.config.decoder_embed_dim = 8192
            self.config.decoder_ffn_embed_dim = 16384
            self.config.decoder_layers = 64
            self.config.decoder_retention_heads = 32
        self.config.block_size = self.args.block_size
        self.config.n_layer = self.config.decoder_layers
        self.config.n_embd = self.config.decoder_embed_dim
        print(self.config.__dict__)

        embed_tokens = Embedding(self.config.vocab_size, self.config.decoder_embed_dim)
        self.model = RetNetDecoder(self.config, embed_tokens=embed_tokens)
        print(self.model)

    def max_positions(self):
        return self.config.max_target_positions

    def forward(self,
                input_ids: torch.LongTensor = None,
                labels: Optional[torch.LongTensor] = None,
                ): #-> Union[Tuple, CausalLMOutput]:
        logits, aux_dic = self.model(input_ids)
        return logits
        # loss = None
        # if labels is not None:
        #     loss = chunked_cross_entropy(logits, labels, chunk_size=self.config.recurrent_chunk_size)
        #
        # return CausalLMOutput(
        #         loss=loss,
        #         logits=logits,
        # )
