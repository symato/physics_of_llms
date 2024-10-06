# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.

import json
import math
import logging
import os, sys
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import LabelSmoother
from mydataset import make_supervised_data_module 
from dataclasses import dataclass, field
from liger_kernel.transformers import AutoLigerKernelForCausalLM

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="../Qwen2.5-1.5B-Instruct__extend_vocab")
    finetune_layers: str = field(default="", metadata={"help": "'0 1 2' ..."})

@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": ".jsonl or .jsonl.xz data filename"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_8bit")
    model_max_length: int = field(
        default=1024*4, # 4k ctxlen
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

local_rank = training_args.local_rank
device_map = None

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

########################################################
####### BEGIN (code thêm vào cho mode PREPARE_DATA only)
PREPARE_DATA_ONLY = ( os.getenv("PREPARE_DATA", 0) == "1" or os.getenv("PREPARE_DATA_ONLY", 0) == "1" )
if PREPARE_DATA_ONLY: rank0_print(">>> PREPARE_DATA_ONLY START")

if ( not PREPARE_DATA_ONLY ) or ( PREPARE_DATA_ONLY and local_rank == 0 ):
    ## Load tokenizer and data với trường hợp training bình thường (not PREPARE_DATA_ONLY) trên mọi processes
    # còn với PREPARE_DATA_ONLY is True thì chỉ load trên process đầu tiên để chuẩn bị dữ liệu
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ## Load data
    # Dùng `training_args.main_process_first(...):` context manager để đảm bảo xử lý data lần đầu tiên
    # trên main process only rồi ghi ra disk, các processes còn lại sẽ load data đã được xử lý từ đĩa.
    # làm vậy sẽ giúp tăng tốc độ xử lý dữ liệu lên rất nhiều lần khi sử dụng đa GPUs
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        data_module = make_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length, 
            rank0_print=rank0_print,
        )

if True: # PREPARE_DATA_ONLY: # Show some sample data to double check
    RESET  = '\033[0m'
    RED    = '\033[91m'
    GREEN  = '\033[32m'
    YELLOW = '\033[33m'
    CYAN   = '\033[36m'
    _MARK  = f"{YELLOW}_{RESET}"

    import random
    train_dataset = data_module["train_dataset"]
    rank0_print(">>>", train_dataset)

    for idx in random.sample(range(len(train_dataset)), 1):
        for index in [idx]:
            pre_input_ids = train_dataset[index - 1]["input_ids"][-8:]
            pre_text = tokenizer.decode(pre_input_ids)

            input_ids = train_dataset[index]["input_ids"]
            labels    = train_dataset[index]["labels"]
            n_tokens = len(input_ids)

            IGNORE_TOKEN_ID = LabelSmoother.ignore_index
            curr, begin = 0, 0
            text = ""

            while curr < n_tokens:
                if labels[begin] == IGNORE_TOKEN_ID:
                    is_end_of_chunk = ( labels[curr] != IGNORE_TOKEN_ID )
                else:
                    is_end_of_chunk = ( labels[curr] == IGNORE_TOKEN_ID ) or ( curr == n_tokens - 1 ) # last one

                if is_end_of_chunk:
                    chunk = input_ids[begin : curr]
                    _text = tokenizer.decode(chunk)

                    if labels[begin] == IGNORE_TOKEN_ID:
                        _text = f"{_MARK}{RED}{_text}{RESET}{_MARK}"

                    else: # đảm bảo labels đúng :)
                        _text = f"{GREEN}{_text}{RESET}"
                        for x in range(begin, curr):
                            assert input_ids[x] == labels[x]

                    text += _text
                    begin = curr # chuyển sang chunk tiếp theo

                    if curr < n_tokens - 1 and \
                        input_ids[curr  ] == tokenizer.pad_token_id and \
                        input_ids[curr+1] == tokenizer.pad_token_id:
                        break # thoát nếu hết văn bản (gặp 2 padding tokens liền nhau)
                curr += 1

            rank0_print(f"\n[pre]{CYAN}{pre_text}{RESET}[pre]")
            rank0_print(f"==> Sample {index}, len {len(text)}:\n{text}")
            rank0_print(f"==> Sample {index}, {n_tokens} tokens\n\n")

    rank0_print(f">>> TOTAL SAMPLES: {len(train_dataset)}\n")

    if PREPARE_DATA_ONLY:
        rank0_print(f">>> PREPARE_DATA_ONLY DONE")
        sys.exit() # xong nhiệm vụ thoát
####### END (code thêm vào cho mode PREPARE_DATA only)
######################################################

config = transformers.AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
)

# model = transformers.AutoModelForCausalLM.from_pretrained(
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    cache_dir=training_args.cache_dir,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # bắt buộc phải có để hoạt động đc với packed dataset
)

## In the training, we set use_cache=False, use_cache=True only takes effect at inference
model.config.use_cache = False


## Finetune embeddings và layers được chọn
# Tham khảo https://github.com/jondurbin/qlora/blob/e4c20638464e70becc212caa955efea378684473/train.py#L1133
if "all" not in model_args.finetune_layers.lower():
    # First, freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze embeddings and finetune_layers
    model.enable_input_require_grads()

    finetune_layers = [ int(x) for x in model_args.finetune_layers.strip().split() ]
    print(">>> finetune_layers", finetune_layers)
    for idx in finetune_layers:
        for param in model.model.layers[idx].parameters():
            param.requires_grad = True


## Detecting last checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

if last_checkpoint is not None:
    print(f">>> resume training from {last_checkpoint} !!!")


## Start training
trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
trainer.train(resume_from_checkpoint=last_checkpoint) # last_checkpoint is None là train từ đầu
trainer.save_state()

## Final save
# if trainer.args.should_save and trainer.args.local_rank == 0:
#     state_dict = trainer.model.state_dict()
#     trainer._save(training_args.output_dir, state_dict = state_dict)
