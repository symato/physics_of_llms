import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from utils import *
from transformers import AutoTokenizer
from threading import Thread
import time

try:
    sys.argv[1]
    input_files = ["test.jsonl.xz"]
except:
    input_files = glob.glob("../NAM-005_436G_Vi-En-Code/*.lzma")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
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


def get_uniq_tokens(infile):
    outfile = infile + "_count.json"

    try: count = json.load(open(outfile))
    except: count = { "last_line_idx": 0 }

    if "last_line_idx" not in count: # DONE
        return count

    texts = []

    for idx, line in enumerate( lzma.open(infile) ):
        if idx <= count["last_line_idx"]:
            continue

        text = json.loads(line)["text"]
        texts.append( text )

        if idx % 2000 == 1999:
            merge_count(count, count_tokens(texts))
            count["last_line_idx"] = idx

            with open(outfile, "wt") as f:
                f.write(json.dumps(count))

            print(f'get_uniq_token {infile}:{count["last_line_idx"]} ...')
            texts = []


    merge_count(count, count_tokens(texts))
    count.pop("last_line_idx")

    with open(outfile, "wt") as f:
        f.write(json.dumps(count))

    print(f'get_uniq_token {infile} DONE.')
    return json.load(open(outfile))


def get_final_count():
    countfile = "tokens_count.json"

    if not os.path.exists(countfile):
        with Pool( processes = num_procs() ) as pool:
            for _ in pool.imap_unordered(get_uniq_tokens, input_files):
                pass

        count = {}
        for infile in input_files:

            x = get_uniq_tokens(infile)
            merge_count(count, x)

        return count

        with open(countfile, "wt") as f:
            f.write(json.dumps(count))

    return json.load(open(countfile))


count = get_final_count()
print(len(count))
