import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from threading import Thread
import re

from utils import *
from unicode_utils import *
from tokens_check import *

min_count = 0
max_count = 0

try:
    x = re.sub(r'/*$', "", sys.argv[1].strip())
    if re.match(r"\d+", x):
        input_files = "stats_mode"
        min_count = int(x)
    else:
        input_files = glob.glob(f"{x}/*.lzma")
except:
    input_files = ["data/test.jsonl.xz"]
print(input_files)


try:
    max_count = int( sys.argv[2] )
except:
    pass
print(min_count, max_count)


PATH = f"data/{model_path}"
mkdirs(PATH)


###
def ok(x):
    tid, count = x

    if count < min_count:
        return False

    token = tokenizer.decode(int(tid))

    if count >= max_count:
        if contains_unwanted(token):
            return False            
    else:
        # Loại nếu không phải ascii
        if not_ascii(token):
            return False

    return True


def count_tokens(texts):
    count = {}
    for text in texts:
        token_ids = tokenizer.encode(text)

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
    x = infile.split("/")[-1]
    outfile = f"{PATH}/{x}_count.json.xz"

    try: count = json.load(lzma.open(outfile))
    except: count = { "last_line_idx": 0 }

    if os.path.exists(infile) and "last_line_idx" in count: # DONE

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

        count = json.load(lzma.open(outfile))


    if "last_line_idx" in count:
        count.pop("last_line_idx")

    # Loại bỏ unwanted và count < 10 & not_ascii
    for k, v in list(count.items()):
        token = tokenizer.decode(int(k))

        if contains_unwanted(token):
            count.pop(k)

        elif v < 10 and not_ascii(token):
            count.pop(k)

    return count



def get_final_count(input_files):
    if input_files == "stats_mode":
        input_files = glob.glob(f"{PATH}/*_count.json.xz")
        input_files = [ x.replace("_count.json.xz", "") for x in input_files ]

    count = {}
    with Pool( processes = num_procs() ) as pool:
        for x in pool.imap_unordered(get_uniq_tokens, input_files):
            merge_count(count, x)

    return count


print("get_final_count ...")
count = get_final_count(input_files)

tid_count_pairs = [ [k, v] for k, v in count.items() ]
total = len(tid_count_pairs)


def remove_not_ok_pairs(pairs):
    keep = []
    remove = []

    for x in pairs:
        if ok(x): keep.append(x)
        else:   remove.append(x)

    return keep, remove

chunk_size = 1024*2
chunks = [tid_count_pairs[i:i + chunk_size] for i in range(0, len(tid_count_pairs), chunk_size)]

remain_pairs = []
removed = []

print("remove_not_ok_pairs ...")
with Pool( processes = num_procs() ) as pool:
    for keep, remove in pool.imap_unordered(remove_not_ok_pairs, chunks):
        remain_pairs += keep
        removed += remove

print("sort remain_pairs and removed ...")
remain_pairs.sort( key = lambda x: -x[1] )
removed.sort( key = lambda x: -x[1] )

x = \
    removed[        :    100] + \
                [[ "0" , 0 ]] + \
    removed[ -10100 : -10000] + \
                [[ "0" , 0 ]] + \
    removed[  -100 :        ] + \
                [[ "0" , 0 ]]

maxx = 25
spaces = " " * 100

for tid, count in x:
    if count == 0:
        print("\n")
        continue

    if tid != "last_line_idx":
        token = json.dumps(tokenizer.decode(int(tid)), ensure_ascii = False)
        n = len(token)
        print(f"{tid}{spaces[:10 - len(tid)]} {token}{spaces[:maxx - n]}\t{count:10.0f}")

print(f"{len(remain_pairs)} / {tokenizer.vocab_size}")


'''
python3 tokens_count.py 5000 20000

86543 / 151643
'''