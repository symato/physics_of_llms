#!/bin/bash

export PACK_DATA=1

data_path=vi_words_similarity
rm -rf data_cached/$data_path

  # --model_name_or_path "../Qwen2.5-1.5B-Instruct" \                             >>> gpu used 3558122496 memory
  # --output_dir "../Qwen2.5-1.5B-Instruct__extend_vocab__1_embeddings_massage" \ >>> gpu used 3363546112 memory
python finetune.py \
  --model_name_or_path "../Qwen2.5-1.5B-Instruct" \
  --finetune_layers "0 1 2 3 4 5 6 7" \
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
  --booster "" \
  --optim adamw_torch \
  --int8_mixed True \
  # --optim adamw_torch \
  # --optim adamw_8bit \
  # --optim ademamix_8bit \
  # --optim adamw_bnb_8bit \
  # --booster "liger" \

############################################################################################################################

## liger booster <= tốn time lúc đầu để compile kernels?
# 0.46 minutes used for training.
# Peak reserved memory for training = 1.576 GB.

## None
# 0.29 minutes used for training.
# Peak reserved memory for training = 2.546 GB.

############################################################################################################################

  # --model_name_or_path "../qwen7b" \
  # --finetune_layers "0 1 2 3" \
  # --optim adamw_torch \
  # --int8_mixed True \

## bf16 + ademamix_8bit
# {'loss': 0.9171, 'grad_norm': 1.84375, 'learning_rate': 0.0, 'epoch': 0.99}
# 100%|███████████████████████████████| 42/42 [00:35<00:00,  1.19it/s]

## bf16 + adamw_8bit
# {'loss': 1.0794, 'grad_norm': 2.578125, 'learning_rate': 0.0, 'epoch': 1.0}
# 100%|███████████████████████████████| 40/40 [00:33<00:00,  1.23it/s

## bf16 + adamw_torch
# {'loss': 0.8688, 'grad_norm': 1.484375, 'learning_rate': 0.0, 'epoch': 1.0}
# 100%|███████████████████████████████| 43/43 [01:17<00:00,  1.80s/it]

## bf16 + adamw_torch + model.compile()
# 100%|███████████████████████████████| 40/40 [02:24<00:00,  3.62s/it]

## 8bit_mixed + adamw_torch
# {'loss': 0.8453, 'grad_norm': 1.578125, 'learning_rate': 0.0, 'epoch': 1.0}
# 100%|███████████████████████████████| 41/41 [03:23<00:00,  4.35s/it]

# mất khoảng 30s để compile rồi mới chạy


############################################################################################################################


# >>> finetune_layers []
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 3.12 minutes used for training.
# Peak reserved memory = 4.049 GB.
# Peak reserved memory for training = 0.621 GB.
# Peak reserved memory % of max memory = 101.225 %.
# Peak reserved memory for training % of max memory = 15.525 %.

# >>> finetune_layers [all]
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 23.72 minutes used for training.
# Peak reserved memory = 10.338 GB.
# Peak reserved memory for training = 6.91 GB.
# Peak reserved memory % of max memory = 258.45 %.
# Peak reserved memory for training % of max memory = 172.75 %.


#   --finetune_layers "0 1 2 3 4 5 6 7" \
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.428 GB of memory reserved.
# 8.82 minutes used for training.
# Peak reserved memory = 5.707 GB.
# Peak reserved memory for training = 2.279 GB.
# Peak reserved memory % of max memory = 142.675 %.
# Peak reserved memory for training % of max memory = 56.975 %.

## Sau khi áp dụng unsloth gradient checkpointing => có thể nhận thấy Peak reserved memory for training giảm rõ rệt!
#   --finetune_layers "0 1 2 3 4 5 6 7" \
# GPU = NVIDIA GeForce RTX 3050 Ti Laptop GPU. Max memory = 4.0 GB.
# 3.607 GB of memory reserved. # khác biệt là do lần test này dùng model nguyên bản vocab x1.5 lần
# 5.26 minutes used for training.
# Peak reserved memory = 5.123 GB.
# Peak reserved memory for training = 1.516 GB.
# Peak reserved memory % of max memory = 128.075 %.
# Peak reserved memory for training % of max memory = 37.9 %.

## Đổi optim từ adamw_8bit sang ademamix_8bit, Peak reserved memory for training tăng lên kha khá
# Peak reserved memory for training = 1.96 GB.
