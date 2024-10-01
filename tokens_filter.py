import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from utils import *
from transformers import AutoTokenizer

try:
    sys.argv[1]
    input_files = ["test.jsonl.xz"]
except:
    input_files = glob.glob("../NAM-005_436G_Vi-En-Code/*.lzma")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    model_max_length = 1024 * 1024 * 4, # 4m ctxlen should fit a book
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

    if not os.path.exists(outfile):

        print(f"get_uniq_token {infile} ...")

        count = {}
        texts =  [ json.loads(line)["text"] for line in lzma.open(infile) ]

        chunk_size = 1024
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        n = int( num_procs() * 0.5 ) 
        with Pool(processes = n) as pool:
            for x in pool.imap_unordered(count_tokens, chunks):
                merge_count(count, x)

        with open(outfile, "wt") as f:
            f.write(json.dumps(count))

    return json.load(open(outfile))


def get_final_count():
    countfile = "tokens_count.json"

    if not os.path.exists(countfile):

        count = {}

        with Pool(processes = 3) as pool:
            for _ in pool.imap_unordered(get_uniq_tokens, input_files):
                pass

        for infile in input_files:
            x = get_uniq_tokens(infile)
            merge_count(count, x)

        with open(countfile, "wt") as f:
            f.write(json.dumps(count))

    return json.load(open(countfile))


count = get_final_count()
print(len(count))
