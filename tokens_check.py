from transformers import AutoTokenizer

from unicode_utils import *

import subprocess, os, sys

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

try: do_check_lang = sys.argv[1] == "bylang"
except: do_check_lang = False

print("do_check_lang", do_check_lang)

if do_check_lang:
	subprocess.run("rm -rf data/tokens_by_lang", shell = True)
	subprocess.run("mkdir -p data/tokens_by_lang", shell = True)

	def check_check(reg, token):
	    m = regex.findall(reg, token)
	    for x in m:
	        for c in x:
	            if ord(c) > 255: # not ascii
	                return True
	    return False


	def write_to_lang_file(lang, token):
	    filename = f"data/tokens_by_lang/{lang}.txt"
	    with open(filename, "at") as f:
	        f.write(token + "\n")


	def check_lang(token):
	    belongs_to_at_least_one_lang = False

	    for lang, reg in unwanted_lang_re_pairs.items():
	        if check_check(reg, token):
	            belongs_to_at_least_one_lang = True
	            write_to_lang_file(lang, token)

	    if not belongs_to_at_least_one_lang:
	        if contains_cjk(token):
	            write_to_lang_file("CJK", token)
	        else:
	            write_to_lang_file("Others", token)


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
