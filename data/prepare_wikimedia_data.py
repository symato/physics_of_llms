import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

import sys; sys.path.append('../'); from utils_lang import *
import sys; sys.path.append('../'); from utils import *

filenames = glob.glob("wiki*train*.jsonl.xz")

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
	return [ x for x in chunk if detect_lang(x) == "vi" and not contains_unwanted(x) and \
		not contains_junks(x) and vietnamese_syllable_ratio(x) > 0.6 and short_line_ratio(x, 15) < 0.2 ]

junks = """
<div 
</div>
class="
class='
<ref>
http://
https://
/CSS-
(CSS-
""".strip().split("\n")


def contains_junks(x):
	x = x.lower()
	for junk in junks:
		if junk in x: return True
	return False


def short_line_ratio(x, min_line_len):
	lines = re.split(r"[\s\n]*\n[\s\n]*", x)
	count = 0
	for line in lines:
		if len(line) < min_line_len:
			count += 1
	return count / len(lines)


cutoff_patterns = """
Tài liệu tham khảo
Tham khảo
Chú thích
""".strip().split("\n")

final_texts = []
with Pool( processes = n ) as pool:
    for wanted_texts in pool.imap_unordered(get_wanted_lang_texts, chunks):
        final_texts += wanted_texts

for text in final_texts:
	for x in cutoff_patterns:
		text = text.split(f"\n{x}\n")[0]
	print(json.dumps({"text": text}, ensure_ascii = False))


'''

wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikisource__20231201.vi__train-00.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-00b.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-01.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-02.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-03b.jsonl.xz

python3 prepare_wikimedia_data.py 3000 > wikimedia_vi_filtered.jsonl

shuf wikimedia_vi_filtered.jsonl | head -n 20 | jq

wc -l wikimedia_vi_filtered.jsonl

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