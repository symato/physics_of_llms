#!/bin/bash

export PACK_DATA=1

data_path=vi_words_similarity
rm -rf data_cached/$data_path

  # --model_name_or_path "../Qwen2.5-1.5B-Instruct" \                             >>> gpu used 3558122496 memory
  # --output_dir "../Qwen2.5-1.5B-Instruct__extend_vocab__1_embeddings_massage" \ >>> gpu used 3363546112 memory
python finetune.py \
  --model_name_or_path "../Qwen2.5-1.5B-Instruct__extend_vocab" \
  --finetune_layers "0 1 2 3" \
  --data_path "$data_path" \
  --model_max_length 512 \
  --output_dir "../Qwen2.5-1.5B-Instruct__extend_vocab__1_embeddings_massage" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "linear" \
  --report_to "none" \
  --bf16 True \
  --optim adamw_8bit


# >>> finetune_layers []
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 3.12 minutes used for training.
# Peak reserved memory = 4.049 GB.
# Peak reserved memory for training = 0.621 GB.
# Peak reserved memory % of max memory = 101.225 %.
# Peak reserved memory for training % of max memory = 15.525 %.


#   --finetune_layers "0 1 2 3 4 5 6 7" \
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 8.82 minutes used for training.
# Peak reserved memory = 5.707 GB.
# Peak reserved memory for training = 2.279 GB.
# Peak reserved memory % of max memory = 142.675 %.
# Peak reserved memory for training % of max memory = 56.975 %.


# >>> finetune_layers [all]
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 23.72 minutes used for training.
# Peak reserved memory = 10.338 GB.
# Peak reserved memory for training = 6.91 GB.
# Peak reserved memory % of max memory = 258.45 %.
# Peak reserved memory for training % of max memory = 172.75 %.
