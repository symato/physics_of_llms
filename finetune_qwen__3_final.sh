#!/bin/bash

export PACK_DATA=1

data_path=final_finetune
rm -rf data_cached/$data_path

python finetune.py \
  --model_name_or_path "../Qwen2.5-7B-Instruct__trimm_vocab" \
  --finetune_layers "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26" \
  --data_path "$data_path" \
  --model_max_length 4096 \
  --output_dir "../Qwen2.5-1.5B-Instruct__trimm_vocab__final" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 3 \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "linear" \
  --report_to "none" \
  --bf16 True \
  --optim adamw_8bit

# GPU = NVIDIA GeForce RTX 4090. Max memory = 23.52 GB.
# 2.916 GB of memory reserved.
