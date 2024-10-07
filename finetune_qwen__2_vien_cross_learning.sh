#!/bin/bash

export PACK_DATA=1

data_path=wikihow_vien_filtered
rm -rf data_cached/$data_path

python finetune.py \
  --model_name_or_path "../Qwen2.5-1.5B-Instruct__extend_vocab__1_embeddings_massage/checkpoint-40" \
  --finetune_layers "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  --data_path "$data_path" \
  --model_max_length 8192 \
  --output_dir "../Qwen2.5-1.5B-Instruct__extend_vocab__2_vien_cross_learning" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "linear" \
  --report_to "none" \
  --bf16 True \
  --optim adamw_8bit
