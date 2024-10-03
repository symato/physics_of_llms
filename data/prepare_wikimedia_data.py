import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

import sys; sys.path.append('../'); from unicode_utils import *
import sys; sys.path.append('../'); from utils import *

filenames = """
wikipedia__20231101.vi__train-02.jsonl.xz
wikisource__20231201.vi__train-00.jsonl.xz
""".strip().split("\n")

min_chars = int(sys.argv[1])

texts = []
for filename in filenames:
	for line in lzma.open(filename):
		if len(line) > min_chars:
			text = json.loads(line)["text"]
			texts.append( text )

n = num_procs() - 1 # giữ lại 1 thread để làm tác vụ khác

chunk_size = round( len(texts) / n )
chunks = [ texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size) ]

def get_wanted_lang_texts(chunk):
	return [ x for x in chunk if not contains_unwanted(x) ]

final_texts = []
with Pool( processes = n ) as pool:
    for wanted_texts in pool.imap_unordered(get_wanted_lang_texts, chunks):
        final_texts += wanted_texts

for text in final_texts:
	print(json.dumps({"text": text}, ensure_ascii = False))

'''

python3 prepare_wikimedia_data.py 3000 | shuf > wikimedia_vi_filtered.jsonl

cat wikimedia_vi_filtered.jsonl | head -n 10 | jq

du -sh wikimedia_vi_filtered.jsonl

'''
'''
## Chuẩn bị data tiếng Việt wikipedia và wikisource
```sh

wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikisource__20231201.vi__train-00.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-02.jsonl.xz

xz -d wikisource__20231201.vi__train-00.jsonl.xz

xz -d wikipedia__20231101.vi__train-02.jsonl.xz

cat wikisource__20231201.vi__train-00.jsonl | head -n 1000 | tail -n 10 | jq
# Có những text rất ngắn, cần loại bỏ

cat wikipedia__20231101.vi__train-02.jsonl | head -n 1000 | tail -n 10 | jq
# Rất nhiều text ngắn, dạng định nghĩa

cat wikipedia__20231101.vi__train-02.jsonl | wc -l # 322170
cat wikipedia__20231101.vi__train-02.jsonl | awk 'length >= 3000' | wc -l # 6465

cat wikisource__20231201.vi__train-00.jsonl | awk 'length >= 3000' | jq
cat wikipedia__20231101.vi__train-02.jsonl | awk 'length >= 3000' | jq
# => cần loại bỏ unwanted langs
```
'''