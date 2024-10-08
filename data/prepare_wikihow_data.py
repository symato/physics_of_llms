import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

import sys; sys.path.append('../'); from utils_lang import *

inputfile = "wikihow_filtered.jsonl"
min_chars = int(sys.argv[1])
max_chars = int(sys.argv[2])

if not os.path.exists(inputfile):
	# Download file if not exists
	cmd = "wget https://huggingface.co/datasets/Symato/wikihow_vi-en-zh/resolve/main/wikihow_filtered.jsonl"
	subprocess.run(cmd, shell = True)

def remove_citation(x):
	return re.sub(r'\s*\[\s*\d+\s*\]\s*', "", x)

# Lọc lấy content song ngữ Anh Việt
for idx, line in enumerate(open(inputfile, "rt")):
	data = json.loads(line)
	if "en" in data and data["en"] is not None:
		if "wikihow" not in data["vi"].lower(): # loại bỏ ' hôm nay wikihow sẽ hướng dẫn các bạn ...'
			# loại bỏ trường dữ liệu tiếng Trung
			if "cn" in data: data.pop("cn")
			# Loại bỏ citation marks. Ví dụ: [16]
			
			en = remove_citation(data["en"])
			vi = remove_citation(data["vi"])

			if idx % 2 == 0:
				human_value = f"Dịch sang tiếng Việt:\n\n{en}"
				gpt_value = vi
			else:
				human_value = f"Dịch sang tiếng Anh:\n\n{vi}"
				gpt_value = en


			conversations = [
				{"from": "human", "value": human_value, "weight": 1}, # bắt buộc giữ lại
				{"from": "gpt", "value": gpt_value},
			]


			line = json.dumps({"conversations": conversations}, ensure_ascii = False)
			n = len(line)

			if n >= min_chars and n <= max_chars \
				and not contains_unwanted(line):
				print(line)

'''

# Xem trước
python3 prepare_wikihow_data.py 2000 20000 | head -n 10 | jq

# Xuất dữ liệu ra file
python3 prepare_wikihow_data.py 2000 20000 > wikihow_vien_filtered.jsonl

wc -l wikihow_vien_filtered.jsonl
# 2938 samples

du -sh wikihow_vien_filtered.jsonl 
# 94M

Có ~100MB text song ngữ Anh Việt để vỗ về embeddings
Ngôn ngữ wikihow đơn giản, dễ đọc, kiến thức đời thường

'''
