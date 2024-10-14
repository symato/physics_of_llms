#!/bin/bash

export PACK_DATA=1

data_path=final_finetune1
#rm -rf data_cached/$data_path

  # --model_name_or_path "../Qwen2.5-7B-Instruct__trimm_vocab" \  
  # --finetune_layers "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25" \
  # --output_dir "../Qwen2.5-7B-Instruct__trimm_vocab__final1" \
  # --per_device_train_batch_size 1 \
  # --gradient_accumulation_steps 16 \
  # --optim adamw_8bit

python finetune.py \
  --model_name_or_path "../Qwen2.5-1.5B-Instruct__extend_vocab" \
  --finetune_layers "all" \
  --data_path "$data_path" \
  --model_max_length 4096 \
  --output_dir "../Qwen2.5-1.5B-Instruct__extend_vocab_final1" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --logging_steps 2 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "linear" \
  --report_to "wandb" \
  --bf16 True \
  --booster "liger" \
  --optim ademamix_8bit
