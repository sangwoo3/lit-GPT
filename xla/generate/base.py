import json
import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch_xla.core.xla_model as xm
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load
from xla.utils import rank_print


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    # TODO: FSDP has an internal broadcasting issue, so we are forced to have this be of length 1 until it's fixed
    input_pos = torch.tensor([0], device=device)

    xm.mark_step()

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def setup(
    prompt: str = "What food do lamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    precision: str = "bf16-true",
):
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        precision: Indicates the Fabric precision setting to use.
    """
    devices = XLAAccelerator.auto_device_count()
    strategy = XLAFSDPStrategy(auto_wrap_policy={Block}) if devices > 1 else "auto"
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch(main, prompt, num_samples, max_new_tokens, top_k, temperature, checkpoint_dir)


def main(
    fabric: L.Fabric,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    checkpoint_dir: Path,
) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    rank_print(fabric, f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint))
    rank_print(fabric, f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens
    assert max_returned_tokens <= model.config.block_size, (
        max_returned_tokens,
        model.config.block_size,
    )  # maximum rope cache length

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        t = time.perf_counter() - t0

        model.reset_cache()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        rank_print(
            fabric,
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
