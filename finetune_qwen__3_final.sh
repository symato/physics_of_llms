#!/bin/bash

export PACK_DATA=1

data_path=final_finetune
# rm -rf data_cached/$data_path

python finetune.py \
  --model_name_or_path "../Qwen2.5-1.5B-Instruct__trimm_vocab__2_vien_cross_learning" \
  --finetune_layers "all" \
  --data_path "$data_path" \
  --model_max_length 8192 \
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
# {'loss': 1.6427, 'grad_norm': 3.3125, 'learning_rate': 1.0344827586206898e-06, 'epoch': 0.0}
# {'train_steps_per_second': 0.15, 'train_loss': 1.5611213878612613, 'epoch': 3.0}
# 312.04 minutes used for training.
# Peak reserved memory = 10.951 GB.
# Peak reserved memory for training = 8.035 GB.
# Peak reserved memory % of max memory = 46.56 %.
# Peak reserved memory for training % of max memory = 34.162 %.
