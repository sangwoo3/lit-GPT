import sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from functools import partial
from tqdm import tqdm
import numpy as np
import nltk

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id


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


def split_data(data, splitter):
    sentences = splitter.tokenize(data["article"])
    return {'sentences': sentences}


def process_data(data, tokenizer, bos=False, eos=False):
    input_ids = []
    for sentence in data["sentences"]:
        sentence_ids = tokenizer(sentence, truncation=False, add_special_tokens=False)["input_ids"]
        if len(sentence_ids) > 0:
            input_ids.append(sentence_ids)
    # input_ids = tokenizer(data["article"], truncation=False, add_special_tokens=False)["input_ids"]
    if bos:
        if bos_id is None:
            raise NotImplementedError("This tokenizer does not defined a bos token")
        input_ids = [bos_id] + input_ids
    if eos:
        if eos_id is None:
            raise NotImplementedError("This tokenizer does not defined a eos token")
        input_ids = input_ids + [eos_id]
    data["input_ids"] = input_ids
    return data


def build_packed_data(destination_path, prefix, chunk_size, dataset):
    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_token_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
    )

    num_tokens = 0
    for td in tqdm(dataset):
        builder.add_array(np.array(td["input_ids"], dtype=builder.dtype))
        num_tokens += len(td["input_ids"])
    print(f"[finished adding to builder array] - prifix: {prefix}")
    print(f"total processed tokens: {num_tokens}")

    builder.write_reminder()
    print("[finished writing to files]")


data_stream = load_dataset('json', data_files='/data2/swcho_data/code/lit-GPT/data/cnn_sample.jsonl',
                           split='train',
                           # streaming=True
                           )
# print(next(iter(data_stream)))
print(data_stream[0])

data_stream = data_stream.train_test_split(test_size=0.1, shuffle=True)
data_stream['validation'] = data_stream.pop('test')

process_ds_split = partial(split_data, splitter=splitter)
data_stream = data_stream.map(process_ds_split,
                              remove_columns=data_stream['train'].column_names,
                              num_proc=2,
                              batched=True,
                              desc='Splitting...')

print(f'bos token: {tokenizer.bos_token} {tokenizer.bos_token_id}')
process_ds = partial(process_data, tokenizer=tokenizer, bos=True)
# original_columns = list(data_stream.features.keys())  # error
tk_dataset = data_stream.map(process_ds, remove_columns=["article", "highlights", "id"], num_proc=10, desc='cnndm')
# ,
# remove_columns=original_columns)
# tk_dataset_updated = tk_dataset.rename_columns(["article", "highlights", "id"])
# print(list(tk_dataset.take(1)))

# ii = 0
# for tk in tk_dataset:
#     ii += 1
ii = len(tk_dataset['train'])
print(f"total number of train instance: {ii}")

n_tk = 0
for i, tk in enumerate(tqdm(tk_dataset['train'])):
    if i < 1:
        print(i, tk)
    n_tk += len(tk['input_ids'])
print(f'[train] iteration is done: {ii} iter / {n_tk} tokens')

ii = len(tk_dataset['validation'])
print(f"total number of valid instance: {ii}")

block_size = 512

chunk_size_train = n_tk // 3 // block_size * block_size
destination_path = Path('/data2/swcho_data/code/lit-GPT/data/cnndm_sample')
destination_path.mkdir(parents=True, exist_ok=True)
build_packed_data(destination_path, 'cnndm-train',
                  chunk_size_train, tk_dataset['train'])

# shuffled_tk_dataset = tk_dataset.shuffle(buffer_size=ii+1, seed=42)
# train_ds = shuffled_tk_dataset.skip(100)
# valid_ds = shuffled_tk_dataset.take(100)


# ii = 0
# n_tk = 0
# for i, tk in enumerate(tqdm(train_ds)):
#     if i < 1:
#         print(i, tk)
#     ii += 1
#     n_tk += len(tk['input_ids'])
# print(f'[train] iteration is done: {ii} iter / {n_tk} tokens')
#
# ii = 0
# for i, tk in enumerate(tqdm(valid_ds)):
#     if i < 1:
#         print(i, tk)
#     ii += 1
# print(f'[valid] iteration is done: {ii} iter')