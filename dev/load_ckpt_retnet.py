import sys
import time
from pathlib import Path

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.config_retnet import arg_loader
from lit_gpt.retnet import RetNet
from torchscale.architecture.retnet import DecoderLayer
from lit_gpt.utils import get_default_supported_precision, step_csv_logger, lazy_load


args = arg_loader()
precision = args.precision or get_default_supported_precision(training=True)

args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size
assert args.gradient_accumulation_steps > 0

if args.devices > 1:
    strategy = FSDPStrategy(
            auto_wrap_policy={DecoderLayer},
            activation_checkpointing_policy={DecoderLayer},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
    )
else:
    strategy = "auto"

logger = step_csv_logger("out", args.exp_name, flush_logs_every_n_steps=args.log_interval)
fabric = L.Fabric(devices=args.devices,
                  strategy=strategy,
                  precision=precision,
                  loggers=logger,
                  accelerator='gpu',
                  num_nodes=args.num_nodes,
                  )

fabric.print(args)
with fabric.init_module(empty_init=True):
    model = RetNet(args)
    model.apply(model._init_weights)

checkpoint_path = 'out/retnet_3b_redpajama_sample/iter-000199-ckpt.pth'

t0 = time.perf_counter()
with lazy_load(checkpoint_path) as checkpoint:
    fabric.print(checkpoint)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

model.eval()
model = fabric.setup_module(model)
