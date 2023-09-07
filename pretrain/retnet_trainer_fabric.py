import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from lit_gpt.model import GPT, Block, Config

from lit_gpt.config_retnet import arg_loader
from lit_gpt.retnet import RetNet
from torchscale.architecture.retnet import DecoderLayer as RetNetDecoderLayer
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger

data_config = [
    # ("arxiv", 2.5),
    ("book", 4.5),
    # ("c4", 15.0),
    # ("cc", 67.0),
    # ("github", 4.5),
    ("stackexchange", 2.0),
    # ("wikipedia", 4.5),
]


def setup() -> None:
    args = arg_loader()
    precision = args.precision or get_default_supported_precision(training=True)

    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    assert args.gradient_accumulation_steps > 0

    if args.devices > 1:
        strategy = FSDPStrategy(
                auto_wrap_policy={RetNetDecoderLayer},
                activation_checkpointing_policy={RetNetDecoderLayer},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = step_csv_logger("out", args.exp_name, flush_logs_every_n_steps=args.log_interval)
    fabric = L.Fabric(devices=args.devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(main, args)


def main(fabric, args):
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    args.out_dir = Path("out") / args.exp_name
    if fabric.global_rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
            batch_size=args.micro_batch_size,
            block_size=args.block_size,
            fabric=fabric,
            train_data_dir=Path(args.train_data_dir),
            val_data_dir=Path(args.val_data_dir),
            seed=(6060 + fabric.global_rank),
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(6060)  # same seed for every process to init model (FSDP)

    # fabric.print(f"Loading model with {config.__dict__}")
    fabric.print(args)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = RetNet(args)
        # model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": model.config, "iter_num": 0, "step_count": 0}

    if args.resume is True:
        resume = sorted(args.out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, speed_monitor, args)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, speed_monitor, args):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = RetNet(args)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * args.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (args.micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    for state["iter_num"], train_data in enumerate(train_dataloader, state["iter_num"]):
        if state["iter_num"] >= args.max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0: model.config.block_size].contiguous()
        targets = train_data[:, 1: model.config.block_size + 1].contiguous()

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / args.gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
                (state["iter_num"] + 1) * args.micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                flops_per_batch=measured_flops,
                lengths=total_lengths,
        )
        if state["iter_num"] % args.log_interval == 0:
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if val_dataloader is not None and not is_accumulating and state["step_count"] % args.eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, args)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and state["step_count"] % args.save_interval == 0:
            checkpoint_path = args.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, args) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(args.eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0: model.config.block_size].contiguous()
        targets = val_data[:, 1: model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
        batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(str(data_dir / f"{prefix}*"))
        dataset = PackedDataset(
                filenames,
                n_chunks=4,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
                f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
        batch_size: int,
        block_size: int,
        fabric,
        train_data_dir: Path = Path("data/redpajama_sample"),
        val_data_dir: Optional[Path] = None,
        seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=train_data_dir,
            shuffle=True,
            seed=seed,
    )
    val_dataloader = (
        create_dataloader(
                batch_size=batch_size,
                block_size=effective_block_size,
                fabric=fabric,
                data_dir=val_data_dir,
                shuffle=False,
                seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    setup()

    # from jsonargparse import CLI
    #
    # CLI(setup)