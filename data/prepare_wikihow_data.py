import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

import sys; sys.path.append('../'); from unicode_utils import *

inputfile = "wikihow_filtered.jsonl"
min_chars = int(sys.argv[1])

if not os.path.exists(inputfile):
	# Download file if not exists
	cmd = "wget https://huggingface.co/datasets/Symato/wikihow_vi-en-zh/resolve/main/wikihow_filtered.jsonl"
	subprocess.run(cmd, shell = True)

def remove_citation(x):
	return re.sub(r'\s*\[\s*\d+\s*\]\s*', "", x)

# Lọc lấy content song ngữ Anh Việt
for line in open(inputfile, "rt"):
	data = json.loads(line)
	if "en" in data and data["en"] is not None:
		if len(data["vi"]) > min_chars and \
			"wikihow" not in data["vi"].lower(): # loại bỏ ' hôm nay wikihow sẽ hướng dẫn các bạn ...'
			# loại bỏ trường dữ liệu tiếng Trung
			if "cn" in data: data.pop("cn")
			# Loại bỏ citation marks. Ví dụ: [16]
			data["en"] = remove_citation(data["en"])
			data["vi"] = remove_citation(data["vi"])
			line = json.dumps(data, ensure_ascii = False)
			if not contains_unwanted(line):
				print(line)

'''

# Xem trước
python3 prepare_wikihow_data.py 10000 | head -n 10 | jq

# Xuất dữ liệu ra file
python3 prepare_wikihow_data.py 10000 > wikihow_more_filtered.jsonl

wc -l wikihow_more_filtered.jsonl
# 2938 samples

du -sh wikihow_more_filtered.jsonl 
# 94M

Có ~100MB text song ngữ Anh Việt để vỗ về embeddings
Ngôn ngữ wikihow đơn giản, dễ đọc, kiến thức đời thường

'''
