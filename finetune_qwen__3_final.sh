#!/bin/bash

export PACK_DATA=1

data_path=final_finetune2
#rm -rf data_cached/$data_path

python finetune.py \
  --model_name_or_path "../Qwen2.5-7B-Instruct__trimm_vocab" \
  --finetune_layers "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25" \
  --data_path "$data_path" \
  --model_max_length 4096 \
  --output_dir "../Qwen2.5-7B-Instruct__trimm_vocab__final2" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
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

# GPU = NVIDIA GeForce RTX 4090. Max memory = 23.52 GB.
# 2.916 GB of memory reserved.
# {'loss': 1.6427, 'grad_norm': 3.3125, 'learning_rate': 1.0344827586206898e-06, 'epoch': 0.0}
# {'train_steps_per_second': 0.15, 'train_loss': 1.5611213878612613, 'epoch': 3.0}
# 312.04 minutes used for training.
# Peak reserved memory = 10.951 GB.
# Peak reserved memory for training = 8.035 GB.
# Peak reserved memory % of max memory = 46.56 %.
# Peak reserved memory for training % of max memory = 34.162 %.
