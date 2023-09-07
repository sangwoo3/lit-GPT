import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader, IterableDataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from lit_gpt import Config
# from lit_gpt.model import GPT, Block
from lit_gpt.config_retnet import arg_loader
from torchscale.architecture.retnet import DecoderLayer as RetNetDecoderLayer
from lit_gpt.retnet import RetNet
from lit_gpt.speed_monitor import SpeedMonitorCallback, estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, step_csv_logger


class LightningGPTModule(L.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.module: Optional[torch.nn.Module] = None
        self.measured_flops: Optional[int] = None

    def configure_model(self) -> None:
        self.module = RetNet(self.args)
        # self.module.apply(self.module._init_weights)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
                self.module.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay, betas=(self.args.beta1, self.args.beta2), foreach=False
        )

    def on_fit_start(self) -> None:
        trainer = self.trainer
        with torch.device("meta"):
            meta_model = RetNet(self.module.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider setting `self.measured_flops = estimated_flops` instead
            estimated_flops = estimate_flops(meta_model) * self.args.micro_batch_size
            self.print(f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (self.args.micro_batch_size, meta_model.config.block_size))
            self.measured_flops = measure_flops(meta_model, x)
            self.print(f"Measured TFLOPs: {self.measured_flops * trainer.world_size / 1e12:.2f}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not self.args.decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(self.trainer.fit_loop.total_batch_idx, self.args)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main():
    args = arg_loader()
    precision = args.precision or get_default_supported_precision(training=True)

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    assert gradient_accumulation_steps > 0

    args.lr_decay_iters = args.max_iters

    out_dir = Path("out") / args.exp_name
    data_dir = Path("data") / args.exp_name

    if args.devices > 1:
        strategy = FSDPStrategy(
                auto_wrap_policy={RetNetDecoderLayer},
                activation_checkpointing_policy={RetNetDecoderLayer},
                # the argument is not available in the Trainer strategy, but it's the default anyways
                # state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = step_csv_logger("out", args.exp_name, cls=CSVLogger, flush_logs_every_n_steps=args.log_interval)
    speed_monitor = SpeedMonitorCallback(
            length_fn=lambda batch: batch[0].size(1), batch_size=args.micro_batch_size, window_size=50,
            time_unit="seconds"
    )
    model_checkpoint = ModelCheckpoint(dirpath=out_dir, every_n_train_steps=args.save_interval,
                                       save_last=True, verbose=True)
    trainer = L.Trainer(
            devices=args.devices,
            strategy=strategy,
            precision=precision,
            logger=logger,
            callbacks=[speed_monitor, model_checkpoint],
            max_steps=args.max_iters,
            max_epochs=1,
            limit_val_batches=args.eval_iters,
            accumulate_grad_batches=gradient_accumulation_steps,
            log_every_n_steps=args.log_interval,
            val_check_interval=args.eval_interval,
            gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 1e-4 else None,
    )

    L.seed_everything(6060, workers=True)  # same seed for every process to init model (FSDP)

    trainer.print(args)

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # config = Config.from_name(model_name)
    # trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    model = LightningGPTModule(args)
    trainer.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    train_data = Dataset(str(data_dir / "train.bin"), args.block_size)
    val_data = Dataset(str(data_dir / "val.bin"), args.block_size)
    train_dataloader = DataLoader(train_data, batch_size=args.micro_batch_size, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=args.micro_batch_size, num_workers=2)

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter() - t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, block_size: int):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.block_size, (1,)).item()
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
            yield x, y


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

    main()

    # from jsonargparse import CLI
    #
    # CLI(main)