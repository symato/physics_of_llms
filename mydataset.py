import json, lzma, math, logging, os, sys, pprint, glob
from typing import Dict, Optional, List
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

PACK_DATA = ( os.getenv("PACK_DATA", 0) == "1")
if PACK_DATA:
    from packed_dataset import PackedDataset, monkey_patch
    monkey_patch()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
assert IGNORE_TOKEN_ID == -100

####################

def preprocess(sources, tokenizer, max_len):

    tknz_name = tokenizer.__class__.__name__.lower()

    if "qwen" in tknz_name:
        from qwen_vocab import old2new_tid

    elif "gemma" in tknz_name:
        from gemma_vocab import old2new_tid

    else:
        assert False, "Không hỗ trợ"

    def tknz(str):
        token_ids = tokenizer(str, add_special_tokens=False).input_ids
        ## không cần nữa bỏ qua first token nữa vì đã có add_special_tokens=False
        token_ids = [ old2new_tid(x, tokenizer) for x in token_ids ]
        token_ids = [ x for x in token_ids if x is not None ]
        return token_ids

    def add_tokens(input_id, target, tokens, ignore=False):
        if isinstance(tokens, str): tokens = tknz(tokens)
        input_id += tokens
        if ignore: target += [IGNORE_TOKEN_ID]*len(tokens)
        else:      target += tokens
        return input_id, target

    input_ids, targets, texts = [], [], []
    skips_count = 0

    for d in sources:

        if "text" in d:
            text_tokens = tknz(d['text']) + [ tokenizer.eos_token_id ]
            if tokenizer.bos_token_id:
                text_tokens = [ tokenizer.bos_token_id ] + text_tokens
            if PACK_DATA: # text sẽ được packing cùng instructs sau
                while len(text_tokens) > 0:
                    input_ids.append(text_tokens[:max_len])
                    targets  .append(text_tokens[:max_len])
                    text_tokens    = text_tokens[max_len:]
            else: # gắn thành chuỗi texts dài để chặn đoạn padding vào instructs ở đoạn sau
                texts += text_tokens
            continue

        input_id, target = [], []
        im_end = "<|im_end|>" # mặc định cho chatml format
        # im_end = "</s>" # để viet-mistal ko phải học cách kết thúc câu mới khi ko finetune embeddings
        # im_end = tokenizer.eos_token # sẽ tự động là <|im_end|> khi xài qwen-based
        for c in d['conversations']:
            # Bắt lỗi
            try: c['from']
            except: assert False, f"{d}\n\n{c}"

            if c['value'] is None:
                assert False, f"{d}\n\n{c}"
            # END bắt lỗi

            if c['from'] == "system":
                add_tokens(input_id, target, "<|im_start|>system\n", ignore=True)
                add_tokens(input_id, target,             c['value'], ignore=True)
                add_tokens(input_id, target,          f"{im_end}\n", ignore=True)


            elif c['from'] == "human":

                ignore = True # mặc định là không học

                if "weight" in c and c["weight"] == 1:
                    ignore = False # chỉ học khi weight đc gán là 1

                add_tokens(input_id, target,   "<|im_start|>user\n", ignore=True)
                add_tokens(input_id, target,             c['value'], ignore=ignore)
                add_tokens(input_id, target,          f"{im_end}\n", ignore=ignore)
            else:

                assert c['from'] == 'gpt'
                ignore = False # mặc định là học

                if "weight" in c and c["weight"] == 0:
                    ignore = True # chỉ bỏ qua khi weight được gán là 0

                add_tokens(input_id, target, "<|im_start|>assistant\n", ignore=True)
                add_tokens(input_id, target,                c['value'], ignore=ignore)
                add_tokens(input_id, target,             f"{im_end}\n", ignore=ignore)


        if len(input_id) <= max_len:
            ######
            assert len(input_id) == len(target)
            input_ids.append(input_id)
            targets.append(target)
            ######

        else:
            skips_count += 1
            # print(f"!!! Bỏ qua sft sample này vì số tokens của nó > {max_len} ctxlen")


    if not PACK_DATA: # Nhét texts vào cuối instructs
        for threshold in [max_len / 2, max_len / 4, max_len / 8]:
            if len(texts) < 100: break
            for i in range(0, len(input_ids)):
                if len(texts) < 100: break
                remain = max_len - len(input_ids[i])
                if remain > threshold:
                    text_tokens = texts[:remain]
                    texts =     s[remain:]
                    input_ids[i] += text_tokens
                    targets[i] += text_tokens

    paddings_count = 0
    for i in range(0, len(input_ids)):
        remain = max_len - len(input_ids[i])
        paddings_count += remain

        if not PACK_DATA: # Padding luôn
            input_ids[i] += [tokenizer.pad_token_id]*remain
            assert len(input_ids[i]) == max_len

            targets[i] += [IGNORE_TOKEN_ID]*remain
            assert len(targets[i]) == max_len

    r = paddings_count / (len(input_ids)*max_len)
    r  = round(r * 10000) / 100

    print(f"\n>>> Tỉ lệ padding {r}%")
    print(f">>> Số lượng mẫu bỏ qua {skips_count}/{len(input_ids)}")
    return input_ids, targets


#### Song song hóa việc chuyển hóa data
from multiprocessing import Pool
from functools import partial
import os, torch, random, gc
#####
class RandomAccessDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        self.random_orders = [x for x in range(len(input_ids))]
        random.shuffle(self.random_orders)

        if os.getenv("TESTING_RANDOM_ACCESS", 0) == "1":
            print(">>> TESTING_RANDOM_ACCESS")
            for i, idx in enumerate(self.random_orders):
                data = self[i]
                assert input_ids[idx] == data["input_ids"]
                assert labels[idx] == data["labels"]
            print("TESTING_RANDOM_ACCESS DONE!")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        idx = self.random_orders[i]
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self):
        super(SupervisedDataset, self).__init__()

    def prepare(self, sources, tokenizer: PreTrainedTokenizer, max_len: int):
        if os.cpu_count() > 80: num_proc = 80
        else: num_proc = os.cpu_count() - 2

        partial_preprocess = partial(preprocess, tokenizer=tokenizer, max_len=max_len)

        chunk_size = 1024
        chunks = [sources[i:i + chunk_size] for i in range(0, len(sources), chunk_size)]

        print(">>> sources", len(sources))
        print(">>> chunks", len(chunks))
        assert sum([len(x) for x in chunks]) == len(sources)

        import time
        start_time = time.time()

        print("Tiền xử lý dữ liệu instructs...")
        input_ids, labels = [], []
        count = 0
        with Pool(processes=num_proc) as pool:
            for i, l in pool.imap_unordered(partial_preprocess, chunks):
                input_ids += i
                labels += l
                count += 1; print(f"\n{count}/{len(chunks)} DONE")
 
        elapsed_time = time.time() - start_time
        print(f"Tokenization time: {elapsed_time} seconds")

        print(">>> sources", len(sources))
        print(">>> input_ids", len(input_ids))

        # Giải phóng bộ nhớ
        sources = None; chunks = None; gc.collect()

        CUTOFF = int(os.getenv("CUTOFF", -1))
        if not PACK_DATA:
            if CUTOFF > 0:
                print(f"!!! chỉ giữ lại {CUTOFF} samples")
                input_ids = input_ids[:CUTOFF]
                labels = labels[:CUTOFF]
                assert len(input_ids) == CUTOFF
                assert len(labels) == CUTOFF

            print("Biến input_ids thành tensors ...")
            self.input_ids = torch.tensor(input_ids)#.cuda()
            input_ids = None; gc.collect() # Giải phóng bộ nhớ

            print("Biến labels thành tensors ...")
            self.labels = torch.tensor(labels)#.cuda()
            labels = None; gc.collect() # Giải phóng bộ nhớ

            self.attention_mask = None
        else:
            print("Packing instructs...")
            rand_access_data = RandomAccessDataset(input_ids, labels)
            packed = PackedDataset(rand_access_data, tokenizer, pack_length=max_len)
            packed.stat(); gc.collect() # Giải phóng bộ nhớ

            if CUTOFF > 0 and CUTOFF < len(packed): print(f"!!! chỉ giữ lại {CUTOFF} samples")
            else: CUTOFF = len(packed)

            input_ids, labels, attention_mask = [], [], []
            for i in range(0, CUTOFF):
                x = packed[i]
                input_ids     .append(x[     "input_ids"])
                labels        .append(x[        "labels"])
                attention_mask.append(x["attention_mask"])

            self.     input_ids = torch.tensor(     input_ids)#.cuda()
            self.        labels = torch.tensor(        labels)#.cuda()
            self.attention_mask = torch.tensor(attention_mask)#.cuda()
            packed = input_ids = labels = attention_mask = None; gc.collect() # Giải phóng bộ nhớ


    def save(self, cache_path):
        torch.save(self.input_ids.cpu(), os.path.join(cache_path, "input_ids.pt"))
        self.input_ids = None

        torch.save(self.labels.cpu(), os.path.join(cache_path, "labels.pt"))
        self.labels = None

        if PACK_DATA:
            torch.save(self.attention_mask.cpu(), os.path.join(cache_path, "attention_mask.pt"))
            self.attention_mask = None

        # Giải phóng bộ nhớ
        gc.collect(); torch.cuda.empty_cache()


    def load(self, cache_path, tokenizer):
        self.input_ids = torch.load(os.path.join(cache_path, "input_ids.pt"))
        self.labels    = torch.load(os.path.join(cache_path, "labels.pt"))
        assert len(self.input_ids) == len(self.labels)
        if PACK_DATA:
            self.attention_mask = torch.load(os.path.join(cache_path, "attention_mask.pt"))
            assert len(self.input_ids) == len(self.attention_mask)

    def __len__(self):
        return len(self.input_ids)

    if not PACK_DATA:
        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
             return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    else:
        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer, training_args, rank0_print = None
) -> Dict:

    """Make dataset and collator for supervised fine-tuning."""
    cache_path = os.path.join("data_cached", training_args.data_path)
    train_dataset = SupervisedDataset()

    if os.path.exists(cache_path):
        print(f">>> {cache_path} existed.")
    else:
        data_path = f"data/{training_args.data_path}.jsonl"
        use_lzma = False
        if not os.path.exists(data_path):
            data_path = f"data/{training_args.data_path}.jsonl.xz"
            use_lzma = True

        print(f"!!! Loading data for supervised finetune from {data_path} ... !!!")
        
        
        if use_lzma: file = lzma.open(data_path, 'rt')
        else:        file =      open(data_path, 'rt')
        sources = [ json.loads(line) for line in file]
        file.close()

        print("Formatting inputs...")
        train_dataset.prepare(
            sources, 
            tokenizer = tokenizer, 
            max_len = training_args.model_max_length,
        )

        print("Save to disk...")
        os.makedirs(cache_path, exist_ok=True)
        train_dataset.save(cache_path)

    train_dataset.load(cache_path, tokenizer)
    x = train_dataset[0]; print("!!!", x)

    if PACK_DATA:
        assert 'attention_mask' in x
        assert x['attention_mask'] is not None
    else:
        assert 'attention_mask' not in x

    return dict(train_dataset=train_dataset, eval_dataset=None)
