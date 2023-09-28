import json
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

from argparse import ArgumentParser

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import TokenizerHF
# from lit_gpt import GPT, Config, Tokenizer
# from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from lit_gpt.config_retnet import arg_loader
from lit_gpt.retnet import RetNet
from lit_gpt.transformer import Transformer


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
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x)  # , max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def main(args) -> None:
    precision = args.precision or get_default_supported_precision(training=False)

    if args.model_type == 'retnet':
        from torchscale.architecture.retnet import DecoderLayer
    elif args.model_type == 'transformer':
        from torchscale.architecture.decoder import DecoderLayer

    if args.strategy == "fsdp":
        args.strategy = FSDPStrategy(auto_wrap_policy={DecoderLayer}, cpu_offload=False)
    fabric = L.Fabric(devices=args.devices, precision=precision, strategy=args.strategy)
    fabric.launch()

    checkpoint_dir = Path(args.checkpoint_dir)
    # check_valid_checkpoint_dir(checkpoint_dir)
    checkpoint_path = sorted(checkpoint_dir.glob("*.pth"))[-1]
    fabric.print(f"checkpoint: {str(checkpoint_path)}")

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        if args.model_type == 'retnet':
            model = RetNet(args)
        elif args.model_type == 'transformer':
            model = Transformer(args)
        config = model.config.__dict__
        fabric.print(f"Model configuration {config}")
        fabric.print(model.model)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint:
        fabric.print(checkpoint.keys())
        model_ckpt = checkpoint.get("model", checkpoint)
        # fabric.print(model_ckpt.keys())
        # model_ckpt = {'.'.join(k.split('.')[1:]): v for k, v in model_ckpt.items()}
        model.load_state_dict(model_ckpt, strict=True)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # model = fabric.setup(model)
    # optimizer = torch.optim.AdamW(
    #         model.parameters(), lr=args.learning_rate,
    #         weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), foreach=False
    # )
    # optimizer = fabric.setup_optimizers(optimizer)
    #
    # state = {"model": model, "optimizer": optimizer, "hparams": model.config, "iter_num": 0, "step_count": 0}
    # t0 = time.perf_counter()
    # fabric.load(str(checkpoint_path), state)
    # fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = TokenizerHF(args.hf_dir)
    encoded = tokenizer.encode(args.prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + args.max_new_tokens
    assert max_returned_tokens <= model.config.block_size, (
        max_returned_tokens,
        model.config.block_size,
    )  # maximum rope cache length

    L.seed_everything(1234)
    for i in range(args.num_samples):
        t0 = time.perf_counter()
        y = generate(
                model,
                encoded,
                max_returned_tokens,
                max_seq_length=max_returned_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
        )
        t = time.perf_counter() - t0

        model.reset_cache()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
                f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
                file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


def add_inference_args(parser):
    parser = ArgumentParser(parents=[parser], add_help=False,
                            description="Generates text samples based on a pre-trained model and tokenizer.")
    parser.add_argument("--prompt", type=str, default="Hello, my name is",
                        help="The prompt string to use for generating the samples.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="The number of text samples to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="The number of generation steps to take.")
    parser.add_argument("--top_k", type=int, default=200,
                        help="The number of top most probable tokens to consider in the sampling process.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="A value controlling the randomness of the sampling process. "
                             "Higher values result in more random samples.")
    parser.add_argument("--checkpoint_dir", type=str, default="",
                        help="The checkpoint directory to load.")
    parser.add_argument("--quantize", type=str, nargs='+',
                        help="Whether to quantize the model and using which method: NOT USED"
                             "- bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes "
                             "- bnb.int8: 8-bit quantization from bitsandbytes "
                             "- gptq.int4: 4-bit quantization from GPTQ "
                             "for more details, "
                             "see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md")
    parser.add_argument("--strategy", type=str, default="auto",
                        help="Indicates the Fabric strategy setting to use.")
    parser.add_argument("--devices", type=int, default=1,
                        help="How many devices to use.")
    # parser.add_argument("--precision", type=str,
    #                     help="Indicates the Fabric precision setting to use.")
    return parser


if __name__ == "__main__":
    # from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
            # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
            "ignore",
            message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    # CLI(main)

    # training arguments
    parser = ArgumentParser(description="Inference RetNet")
    parser = arg_loader(parser)
    parser = add_inference_args(parser)
    args = parser.parse_args()

    main(args)
