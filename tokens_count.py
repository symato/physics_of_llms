import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

from utils import *
from utils_lang import *
from tokens_check import *

min_count = 0
max_count = 0

try:
    # bỏ / ở cuối tham số đầu vào
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


from config import ONLINE_MODEL_PATH as model_path
PATH = f"data/{model_path}"
mkdirs(PATH)


latin_tids = [ json.loads(line)["tid"] for line in lzma.open("data/tokens_by_lang/Latin.jsonl.xz", "rt") ]
latin_tids = set(latin_tids)
###
def ok(x):

    tid, count = x
    tid = int(tid)

    if tid in latin_tids:
        return False

    token = tokenizer.decode(tid)

    if contains_unwanted(token):
        return False

    if count < min_count:

        if contains_emoji(token):
            return True

        if is_alphabet(token):
            return True

    elif count < max_count:

        if canbe_vietnamese(token):
            return True

        if contains_emoji(token):
            return True

        if is_ascii(token):
            return True

    else:
        return True

    return False


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

kept = []
removed = []

print("remove_not_ok_pairs ...")
with Pool( processes = num_procs() ) as pool:
    for keep, remove in pool.imap_unordered(remove_not_ok_pairs, chunks):
        kept += keep
        removed += remove

print("sort kept pairs and removed pairs ...")
kept.sort( key = lambda x: -x[1] )
removed.sort( key = lambda x: -x[1] )
mid = len(removed) // 2
x = \
    removed[        :    100] + \
                [[ "0" , 0 ]] + \
    removed[ mid-50 : mid+50] + \
                [[ "0" , 0 ]] + \
    removed[  -100 :        ] + \
                [[ "0" , 0 ]]

maxx = 25
spaces = " " * 100

def pretty(tid, count):
    token = json.dumps(tokenizer.decode(int(tid)), ensure_ascii = False)
    n = len(token)
    return f"{tid}{spaces[:10 - len(tid)]} {token}{spaces[:maxx - n]}\t{count:10.0f}"


print("\n=== Một số removed tokens ===\n")
for tid, count in x:

    if count == 0:
        print("\n")
    else:
        print(pretty(tid, count))


def pretty_token(token, tid, count):
    s = json.dumps([ token, tid, count ], ensure_ascii = False)
    s = "[  " + s[1:]
    a, b = s.split(", ", 1)
    return f"{a}{spaces[:50 - len(a)]}, {b}" + "\n"


def pretty_json(tid, count):
    tid = int(tid)
    token = tokenizer.decode(tid)
    return pretty_token(token, tid, count)


subprocess.run("rm tokens_*.jsonl", shell = True)


for tid, count in removed:
    tid = int(tid)
    token = tokenizer.decode(tid)

    p_token = pretty_token(token, tid, count)
    if is_ascii(token):
        if is_alphabet(token):
            if is_english_word(token):
                with open("tokens_kept__english.jsonl", "at") as f:
                    f.write(p_token)
            else:
                with open("tokens_removed__alphabet.jsonl", "at") as f:
                    f.write(p_token)
        else:
            with open("tokens_removed__ascii.jsonl", "at") as f:
                f.write(p_token)
    else:
        with open("tokens_removed__others.jsonl", "at") as f:
            f.write(p_token)


for tid, count in kept:
    tid = int(tid)
    token = tokenizer.decode(tid)

    p_token = pretty_token(token, tid, count)
    if is_ascii(token):
        if is_alphabet(token):
            if is_english_word(token):
                with open("tokens_kept__english.jsonl", "at") as f:
                    f.write(p_token)
            else:
                if len(token) > 12:
                    with open("tokens_kept__alphabet_long.jsonl", "at") as f:
                        f.write(p_token)
                else:
                    with open("tokens_kept__alphabet_short.jsonl", "at") as f:
                        f.write(p_token)
        else:
            if len(token) > 12:
                with open("tokens_kept__ascii_long.jsonl", "at") as f:
                    f.write(p_token)
            else:
                with open("tokens_kept__ascii_short.jsonl", "at") as f:
                    f.write(p_token)
    else:
        with open("tokens_kept__others.jsonl", "at") as f:
            f.write(p_token)


remains = set(wanted_tids)
for tid, _ in removed:
    tid = int(tid)
    if tid in remains:
        remains.remove(tid)


print(f"kept    / total = {len(kept)} / {tokenizer.vocab_size}")
print(f"remains / total = {len(remains)} / {tokenizer.vocab_size}")
print("( remains = wanted - removed )")

"""

python3 tokens_count.py 1000 20000

kept    / total = 79719 / 151643
remains / total = 80147 / 151643
( remains = wanted - removed )

Final = kept + special tokens

"""
