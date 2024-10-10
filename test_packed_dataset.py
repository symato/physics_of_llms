#!/usr/bin/env python3
import os; os.environ["PACKED_PATCH_CHECK"] = "1"; os.environ["PACK_DATA"] = "1"
import torch, math, gc, sys
import transformers, peft
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from packed_dataset import PackedDataset, monkey_patch, get_unpad_data

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_mistral

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import random, os

## Config
FAST_TEST = os.getenv("fast", 0) == "1"
LEN = int(os.getenv("len", 256))

@dataclass
class InfoOnlyTokenizer:
    """Fake Tokenizer"""
    padding_side="right"
    pad_token_id=2 # eos_token_id


class RandomDataset(Dataset):
    def __init__(self, tokenizer: Any, pack_length: int, length: int) -> None:
        super().__init__()
        self.pack_length = pack_length
        self.tokenizer = tokenizer
        self.length = length
        self.input_ids = [ np.random.randint(16000, size=random.randint(3, int(pack_length//1.5))).tolist() \
            for i in range(length) ]

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        x = self.input_ids[i]
        labels = x
        return dict(input_ids=x, labels=labels)


class MyDataset(Dataset):
    def __init__(self, raw_dataset: RandomDataset) -> None:
        super().__init__()
        self.pack_length = raw_dataset.pack_length
        self.tokenizer = raw_dataset.tokenizer
        self.input_ids = raw_dataset.input_ids
        self.length = raw_dataset.length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        pad_length = self.pack_length - len(self.input_ids[i])
        input_ids = self.input_ids[i] + [self.tokenizer.pad_token_id]*pad_length
        labels = self.input_ids[i] + [LabelSmoother.ignore_index]*pad_length
        labels[0] = LabelSmoother.ignore_index
        # This is to make sure that the first token won't be included in computing loss

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        # return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(tokenizer.pad_token_id))
        return dict(input_ids=input_ids, labels=labels)

############################################################################################

tokenizer = InfoOnlyTokenizer()

raw_dataset = RandomDataset(tokenizer, pack_length=256, length=LEN)
my_dataset = MyDataset(raw_dataset)

packed_dataset = PackedDataset(raw_dataset, tokenizer, raw_dataset.pack_length, return_tensor=True)
packed_dataset.stat()

if LEN > 512:
    print("set LEN <= 512 để chạy code test phần flash_attention_2 monkey patch")
else:
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None # This argument serves for adding new tokens.
    )
    def load_model(model_name, booster = None, use_lora = False):
        if booster == "liger":
            from_pretrained = AutoLigerKernelForCausalLM.from_pretrained
        else:
            from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

        if booster is None or booster == "liger":
            model = from_pretrained(
                model_name,
                trust_remote_code = False,
                use_cache = False,
                torch_dtype = torch.bfloat16,
                quantization_config = None,
                attn_implementation = "flash_attention_2", # bắt buộc phải có để dùng packed dataset
            )
            model.to('cuda')

            if use_lora:
                model = peft.get_peft_model(model, lora_config)
        else:
            assert False, f"Not implemented booster {booster}"

        return model

    ############################################################################################

    def compute_loss_of_model(model: Any, ds: Dataset, batch_size=4):
        data_loader = DataLoader(ds, batch_size, shuffle=False)
        total_loss = 0
        total_num_loss_tokens = 0  # this is the total number of tokens for computing loss

        for index, batch in enumerate(data_loader):
            # print(f"compute loss for batch: {index}, {batch}")
            for key in batch:
                batch[key] = batch[key].to(model.device)
            batch["return_dict"] = True

            with torch.no_grad():
                avg_loss = model.forward(**batch).loss.item()
                # compute number of tokens used for computing loss
                labels = batch["labels"]
                shift_labels = labels[..., 1:].contiguous().view(-1)
                ignore_count = (shift_labels == LabelSmoother.ignore_index).sum()
                num_tokens = shift_labels.size(0) - ignore_count

                total_num_loss_tokens += num_tokens.item()
                total_loss += avg_loss * num_tokens.item()

        return total_loss, total_num_loss_tokens


    ############################################################################################
    ORIGINAL_GET_UNPAD_DATA = transformers.modeling_flash_attention_utils._get_unpad_data

    try:    model_name = sys.argv[1]
    except: model_name = "Qwen/Qwen1.5-0.5B"
    print(model_name)#; input()

    results = []
    for booster in [None, "liger"]:
        # booster như unsloth, liger sẽ sửa đổi code của transformers nên None cần đc chạy trước
        for use_lora in [False, True]:
            print("$$$$", model_name, booster, use_lora)
            model = load_model(model_name, booster = booster, use_lora = use_lora)

            transformers.modeling_flash_attention_utils._get_unpad_data = ORIGINAL_GET_UNPAD_DATA 
            x = compute_loss_of_model(model, my_dataset)

            monkey_patch(caller="test_packed_dataset.py")
            y = compute_loss_of_model(model, packed_dataset)

            results.append(f"\n>>> booster {booster}, lora {use_lora}")

            avg_loss, token_count = x
            packed_avg_loss, packed_token_count = y

            assert token_count == packed_token_count, "number of tokens for computing loss is different: "\
                f"original_token_count = {token_count}, mk_token_count={packed_token_count}"

            diff_loss = math.fabs(packed_avg_loss - avg_loss) / avg_loss
            results[-1] += f"\noriginal_loss: {avg_loss}, multipacking_loss: {packed_avg_loss}, diff={diff_loss*100:2.4f}%"

            for i in [0, 2, 1, 3]:
                if i < len(results):
                    print(results[i])
            print()

            model = None; gc.collect(); torch.cuda.empty_cache()

            if FAST_TEST: break
        if FAST_TEST: break
