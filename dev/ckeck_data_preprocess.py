from datasets import load_dataset
from transformers import AutoTokenizer

from functools import partial
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id


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


data_stream = load_dataset('json', data_files='/data2/swcho_data/code/lit-GPT/data/cnn_sample.jsonl',
                           split='train', streaming=True)
next(iter(data_stream))

process_ds = partial(process_data, tokenizer=tokenizer, bos=True)
tk_dataset = data_stream.map(process_ds)
next(iter(tk_dataset))

ii = 0
for i, tk in enumerate(tqdm(tk_dataset)):
    if i < 3:
        print(i, tk)
    ii += 1
print(f'iteration is done {ii} iter')