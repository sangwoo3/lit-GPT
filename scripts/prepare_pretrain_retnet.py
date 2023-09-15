# import glob
# import json
# import os
import sys
from pathlib import Path

import numpy as np
# from tqdm import tqdm

from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
# from lit_gpt import Config, Tokenizer

tokenizer_dir = "/apdcephfs/share_300000800/user/swcho/huggingface_models/"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
bos_id = tokenizer.bos_id
eos_id = tokenizer.eos_id


def process_data(data, tokenizer, bos=False, eos=False):
    input_ids = tokenizer(data["text"], truncation=False, add_special_tokens=False)["input_ids"]
    if bos:
        if bos_id is None:
            raise NotImplementedError("This tokenizer does not defined a bos token")
        input_ids = [bos_id] + input_ids
    if eos:
        if eos_id is None:
            raise NotImplementedError("This tokenizer does not defined a eos token")
        input_ids = input_ids + [eos_id]
    # data["input_ids"] = input_ids
    return input_ids


def process(source_path: Path,
            destination_path: Path,
            prefix: str,
            chunk_size: int,
            num_proc: int):
    source_file = str(source_path / "merged_360G_v2.jsonl")

    merged_dataset_streamed = load_dataset(
            "json", data_files=source_file, streaming=True
    )
    print(next(iter(merged_dataset_streamed)))

    process_dataset = partial(process_data, tokenizer=tokenizer, bos=True)
    tokenized_dataset = merged_dataset_streamed.map(process_dataset, num_proc=num_proc)
    print(next(iter(tokenized_dataset)))

    destination_path.mkdir(parents=True, exist_ok=True)

    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
    )

    # TODO: test with small data
    for td in tokenized_dataset:
        builder.add_array(np.array(td, dtype=builder.dtype))

    builder.write_reminder()


def prepare(
        source_path: Path = Path("/apdcephfs/share_300000800/user/riversong/pile/train"),
        destination_path: Path = Path("/apdcephfs/share_300000800/user/swcho/data/pretrain"),
        block_size: int = 2048,
        prefix: str = "PCS-merged",
) -> None:
    block_size = block_size

    max_cpus = multiprocessing.cpu_count()
    num_proc = int(max_cpus * 0.9)

    process(
            source_path=source_path,
            destination_path=destination_path,
            prefix=prefix,
            chunk_size=(block_size + 1) * 1024 * 1024,  # block size + 1 for causal, 1024 * 1024 blocks
            num_proc=num_proc,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)