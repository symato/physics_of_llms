#!/bin/bash

export PACK_DATA=1

data_path=wikihow_vien_filtered
rm -rf data_cached/$data_path

python finetune.py \
  --model_name_or_path "../gemma-2-9b-it__trimm_vocab" \
  --finetune_layers "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  --data_path "$data_path" \
  --model_max_length 8192 \
  --output_dir "../gemma-2-9b-it__trimm_vocab__2_vien_cross_learning" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 3 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "linear" \
  --report_to "none" \
  --bf16 True \
  --optim adamw_8bit


# >>> finetune_layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# GPU = NVIDIA GeForce RTX 4090. Max memory = 23.52 GB.
# 2.916 GB of memory reserved.
# 44.63 minutes used for training.
# Peak reserved memory = 7.539 GB.
# Peak reserved memory for training = 4.623 GB.
# Peak reserved memory % of max memory = 32.054 %.
# Peak reserved memory for training % of max memory = 19.656 %.
