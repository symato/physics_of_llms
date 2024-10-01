import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from utils import *
from transformers import AutoTokenizer
from threading import Thread
import re

try:
    x = re.sub(r'/*$', "", sys.argv[1].strip())
    input_files = glob.glob(f"{x}/*.lzma")
except:
    input_files = ["data/test.jsonl.xz"]

try:    model_path = sys.argv[2]
except: model_path = "Qwen/Qwen2.5-14B-Instruct"


PATH = f"data/{model_path}"
mkdirs(PATH)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length = 1024 * 1024 * 4, # 4m ctxlen có thể chứa 1 cuốn sách
)


def count_tokens(texts):
    count = {}
    for text in texts:
        # tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        # text = tokenizer.decode(token_ids)

        for tid in token_ids:

            if tid not in count:
                count[tid] = 0

            count[tid] += 1
    return count



def merge_count(count, x):
    for k, v in x.items():

        if k not in count:
            count[k] = 0

        count[k] += v


def get_uniq_tokens(infile, fast = False):
    x = infile.split("/")[-1]
    outfile = f"{PATH}/{x}_count.json.xz"

    try: count = json.load(lzma.open(outfile))
    except: count = { "last_line_idx": 0 }

    if fast or "last_line_idx" not in count: # DONE
        return count

    texts = []

    for idx, line in enumerate( lzma.open(infile) ):
        if idx <= count["last_line_idx"]:
            continue

        text = json.loads(line)["text"]
        texts.append( text )

        if idx % 10000 == 9999:
            merge_count(count, count_tokens(texts))
            count["last_line_idx"] = idx

            with lzma.open(outfile, "wt") as f:
                f.write(json.dumps(count))

            print(f'get_uniq_token {infile}:{count["last_line_idx"]} ...', flush = True)
            texts = []


    merge_count(count, count_tokens(texts))
    count.pop("last_line_idx")

    with lzma.open(outfile, "wt") as f:
        f.write(json.dumps(count))

    print(f'get_uniq_token {infile} DONE.', flush = True)
    return json.load(lzma.open(outfile))


def get_final_count(fast = False):
    countfile = f"{PATH}/tokens_count.json.xz"

    if not os.path.exists(countfile):
        with Pool( processes = num_procs() ) as pool:
            for _ in pool.imap_unordered(get_uniq_tokens, input_files):
                pass

        count = {}
        for infile in input_files:

            x = get_uniq_tokens(infile, fast = fast)
            merge_count(count, x)

        return count

        with lzma.open(countfile, "wt") as f:
            f.write(json.dumps(count))

    return json.load(lzma.open(countfile))


count = get_final_count(fast = True)
print(len(count))
