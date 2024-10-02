from transformers import AutoTokenizer

from unicode_utils import *

import subprocess, os

model_path = "Qwen/Qwen2.5-14B-Instruct"
# model_path = "meta-llama/Llama-3.1-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length = 1024 * 1024 * 4, # 4m ctxlen có thể chứa 1 cuốn sách
)


# if __name__ == "__main__":

wanted = open("data/tokens_wanted.txt", "wt")
unwanted = open("data/tokens_unwanted.txt", "wt")

wanted_tids = []

try: do_check_lang = sys.argv[1]
except: do_check_lang = False

if do_check_lang:
	subprocess.run("rm -rf data/langs", shell = True)
	subprocess.run("mkdir -p data/langs", shell = True)

for tid in range(0, tokenizer.vocab_size):
	token = tokenizer.decode(tid)

	if do_check_lang:
		check_lang(token)

	if contains_unwanted(token):
		unwanted.write(token + "\n")
	else:
		wanted_tids.append(tid)
		wanted.write(token + "\n")

print(f"wanted {len(wanted_tids)} / {tokenizer.vocab_size}")
