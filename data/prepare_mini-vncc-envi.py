import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

import sys; sys.path.append('../'); from utils_lang import *

inputfile = "mini-vncc-envi_30k.jsonl.xz"

# Lọc lấy content song ngữ Anh Việt
for idx, line in enumerate(lzma.open(inputfile, "rt")):
	data = json.loads(line)

	vi = data["vi"] # sưu tầm từ internet
	en = data["en"] # là bản dịch google từ vi

	if vietnamese_syllable_ratio(vi) > 0.73:

		value = f"Dịch sang tiếng Việt:\n\n{en}"

		conversations = [
			{"from": "human", "value": value, "weight": 0}, # vì thế nên không huấn luyện trên data dịch
			{"from": "gpt",   "value": vi,    "weight": 1}, # mà chỉ huấn luyện trên data gốc
		]

		line = json.dumps({"conversations": conversations}, ensure_ascii = False)

		print(line)
