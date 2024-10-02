from transformers import AutoTokenizer

from unicode_utils import *

model_path = "Qwen/Qwen2.5-14B-Instruct"
# model_path = "meta-llama/Llama-3.1-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length = 1024 * 1024 * 4, # 4m ctxlen có thể chứa 1 cuốn sách
)

wanted = open("tokens_wanted.txt", "wt")
unwanted = open("tokens_unwanted.txt", "wt")

wanted_tids = []

for tid in range(0, tokenizer.vocab_size):
	token = tokenizer.decode(tid)

	if contains_unwanted(token):
		unwanted.write(token + "\n")
	else:
		wanted_tids.append(tid)
		wanted.write(token + "\n")

print(f"wanted {len(wanted_tids)} / {tokenizer.vocab_size}")
