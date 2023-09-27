from transformers import AutoTokenizer
import torch
from typing import Optional


class TokenizerHF:
    def __init__(self, tokenizer_dir: str) -> None:
        self.processor = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.bos_id = self.processor.bos_token_id
        self.eos_id = self.processor.eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.processor)

    def token_to_id(self, token: str) -> int:
        id_ = self.processor.token_to_id(token)
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(
            self,
            string: str,
            device: Optional[torch.device] = None,
            bos: bool = False,
            eos: bool = False,
            max_length: int = -1,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)