# import glob
# import json
# import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing
import nltk

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

tokenizer_dir = "/apdcephfs/share_300000800/user/swcho/huggingface_models/"
if True:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
else:
    tokenizer = Tokenizer(Path(tokenizer_dir))
    eos_id = tokenizer.eos_id

n_train_files = 512
n_valid_files = 32
block_size = 2048
n_tk_train_est = 100000000000
n_tk_valid_est = 10000000


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


nltk_splitter = nltk.load('tokenizers/punkt/english.pickle')
splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
        train_text=nltk_splitter._params,
        lang_vars=CustomLanguageVars()
)


def split_data(data):
    if not isinstance(data["text"], str):
        text = ' '.join(data["text"])
    else:
        text = data["text"]
    sentences = splitter.tokenize(text)
    return {'sentences': sentences}


def process_data(data, tokenizer, bos=False, eos=False):
    doc_ids = []
    for sentence in data['sentences']:
        sentence_ids = tokenizer(sentence, truncation=False, add_special_tokens=False)["input_ids"]
        if len(sentence_ids) > 0:
            doc_ids.append(sentence_ids)
    # doc_ids = tokenizer(data["text"], truncation=False, add_special_tokens=False)["input_ids"]
    if bos:
        if bos_id is None:
            raise NotImplementedError("This tokenizer does not defined a bos token")
        doc_ids = [bos_id] + doc_ids
    if eos:
        if eos_id is None:
            raise NotImplementedError("This tokenizer does not defined a eos token")
        doc_ids = doc_ids + [eos_id]
    return {"input_ids": doc_ids}


def build_packed_data(destination_path, prefix, chunk_size, dataset):
    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
    )

    # num_tokens = 0
    for td in tqdm(dataset):
        builder.add_array(np.array(td["input_ids"], dtype=builder.dtype))
        # num_tokens += len(td["input_ids"])
    print(f"[finished adding to builder array] - prifix: {prefix}")
    # print(f"total processed tokens: {num_tokens}")

    builder.write_reminder()
    print("[finished writing to files]")


def build_packed_data_sp(destination_path, prefix, chunk_size, dataset):
    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
    )

    num_tokens = 0
    for tds in tqdm(dataset):
        text_ids = tokenizer.encode(tds["text"])
        builder.add_array(np.array(text_ids, dtype=builder.dtype))
        num_tokens += len(text_ids)
    print(f"[finished adding to builder array] - prifix: {prefix}")
    print(f"total processed tokens: {num_tokens}")

    builder.write_reminder()
    print("[finished writing to files]")


def process(source_path: Path,
            destination_path: Path,
            prefix: str,
            num_proc: int):
    source_file = str(source_path / "merged_360G_v2.jsonl")
    merged_dataset = load_dataset(
            "json", data_files=source_file, split='train',
            # streaming=True,
    )
    # print(next(iter(merged_dataset_streamed)))
    print(merged_dataset[0])

    # validation set: 0.01%
    merged_dataset = merged_dataset.train_test_split(test_size=0.0001, shuffle=True)
    merged_dataset['valid'] = merged_dataset.pop('test')

    '''
    # sentencepiece tokenize
    destination_path.mkdir()
    
    chunk_size_train = n_tk_train_est // n_train_files // (block_size + 1) * (block_size + 1)
    chunk_size_valid = n_tk_valid_est // n_valid_files // (block_size + 1) * (block_size + 1)
    print(f"chunk size: train-{chunk_size_train}, valid-{chunk_size_valid}")

    build_packed_data_sp(destination_path, prefix + '-train',
                         chunk_size_train, merged_dataset['train'])
    build_packed_data_sp(destination_path, prefix + '-valid',
                         chunk_size_train, merged_dataset['valid'])
    '''

    # HF tokenize
    t0 = time.time()
    merged_dataset = merged_dataset.map(split_data,
                                        remove_columns=["text"],
                                        num_proc=num_proc,
                                        batched=True,
                                        desc='Splitting...')
    t1 = time.time()
    print(f"[finished splitting] elapsed: {(t1 - t0) * 1000}sec")

    t0 = time.time()
    process_dataset = partial(process_data, tokenizer=tokenizer, bos=True)
    tokenized_dataset = merged_dataset.map(process_dataset,
                                           remove_columns=["sentences"],
                                           num_proc=num_proc,
                                           batched=True,
                                           desc='Tokenizing...')
    t1 = time.time()
    # print(next(iter(tokenized_dataset)))
    print(tokenized_dataset[0])
    print(f"[finished tokenize] elapsed: {(t1 - t0) * 1000}sec")
    print(f"train dataset size: {len(tokenized_dataset['train'])}")
    print(f"valid dataset size: {len(tokenized_dataset['valid'])}")

    n_tk_train, n_tk_valid = 0, 0
    for tk in tqdm(tokenized_dataset['train'], desc='train size...'):
        n_tk_train += len(tk['input_ids'])
    for tk in tqdm(tokenized_dataset['valid'], desc='valid size...'):
        n_tk_valid += len(tk['input_ids'])
    print(f"total token count: train-{n_tk_train}, valid-{n_tk_valid}")

    # n_ds, n_tk = 0, 0
    # for tk in tokenized_dataset:
    #     n_ds += 1
    #     n_tk += len(tk['input_ids'])
    # print(f"total tokenized tokens: {n_tk}, dataset size: {n_ds}")

    # file_size_byte = os.stat(source_file).st_size
    # chunk_size = file_size_byte // num_files // (block_size + 1)  # block size + 1 for causal
    # print(f"source file size: {file_size_byte}bytes, target file count: {num_files}, chunk size: {chunk_size}")

    # shuffled_tk_dataset = tokenized_dataset.shuffle(buffer_size=n_ds_train + 1, seed=42)
    # n_val = int(n_ds * 0.0001)
    # print("suffling is done with tokenized dataset")
    # print(f"valid instance: {n_val} / total instance: {n_ds}")
    # shuffled_ds_train = shuffled_tk_dataset.skip(n_val)
    # shuffled_ds_valid = shuffled_tk_dataset.take(n_val)

    chunk_size_train = n_tk_train // n_train_files // (block_size + 1) * (block_size + 1)
    chunk_size_valid = n_tk_valid // n_valid_files // (block_size + 1) * (block_size + 1)
    print(f"chunk size: train-{chunk_size_train}, valid-{chunk_size_valid}")

    destination_path.mkdir(parents=True, exist_ok=True)

    build_packed_data(destination_path, prefix + '-train',
                      chunk_size_train, tokenized_dataset['train'])

    build_packed_data(destination_path, prefix + '-val',
                      chunk_size_valid, tokenized_dataset['valid'])


def prepare(
        source_path: Path = Path("/apdcephfs/share_300000800/user/riversong/pile/train"),
        destination_path: Path = Path("/apdcephfs/share_300000800/user/swcho/data/pretrain_retnet"),
        prefix: str = "PCS-merged-360G",
) -> None:
    max_cpus = multiprocessing.cpu_count()
    num_proc = max_cpus - 5  # int(max_cpus * 0.9)
    print(f"total cpu count: {max_cpus}, number of cpus to use: {num_proc}")

    process(
            source_path=source_path,
            destination_path=destination_path,
            prefix=prefix,
            num_proc=num_proc,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)