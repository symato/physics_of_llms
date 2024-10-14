import os, sys, glob, json
from utils_lang import *
from transformers import AutoTokenizer

from config import ONLINE_MODEL_PATH as model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_kept_tids():
    # Keep all special tokens
    kept_tids = set( x for x in range(151643, 151664 + 1) )

    canbe_vi_kept = 0
    is_ascii_kept = 0

    for tid in range(0, tokenizer.vocab_size):
        token = tokenizer.decode(tid)

        if vietnamese_syllable_ratio(token) > 0.8:
            canbe_vi_kept += 1
            kept_tids.add(tid)

        if len(token) <= 2 and canbe_vietnamese(token):
            canbe_vi_kept += 1
            kept_tids.add(tid)            

        if len(token) <= 2 and is_ascii(token):
            is_ascii_kept += 1
            kept_tids.add(tid)

    print(">>> canbe_vi_kept", canbe_vi_kept)
    print(">>> is_ascii_kept", is_ascii_kept)
    # '''

    kept_filenames = glob.glob("qwen__1000__20000/tokens_kept__*.jsonl")

    for filename in kept_filenames:
        for line in open(filename, "rt"):
            token, tid, count = json.loads(line)
            kept_tids.add(tid)

    kept_tids = list( kept_tids )
    kept_tids.sort()

    print("new_qwen_vocab", len(kept_tids))
    return kept_tids


kept_tids = get_kept_tids()

# old vs new vocab mapping
old2new = {}
new2old = {}

for new_tid, old_tid in enumerate( kept_tids ):
    old2new[ old_tid ] = new_tid
    new2old[ new_tid ] = old_tid


STRANGE_TOKENS = set()

def old2new_tid(x, tokenizer):
    global STRANGE_TOKENS

    if x in old2new:
        return old2new[x]

    else:
        token = tokenizer.decode(x)
        if contains_unwanted(token):
            return None

        words = re.findall(r'[a-z]+', token, flags = re.IGNORECASE)

        if len(words) > 1:
            print(">>>", words)

        if len(words) == 1:
            tids = tokenizer.encode(words[0])
            if len(tids) == 1 and tids[0] in old2new:
                return old2new[tids[0]]

        msg = f">>> old2new_tid error: id {x}, token '{token}'"
        if token not in STRANGE_TOKENS:
            print(msg)
            STRANGE_TOKENS.add( token )

        # assert False, msg
        return None

    assert False, "Không thể tới bước này, có lỗi ở phần code trên"


from pyvi import ViTokenizer
new2tid = json.load(open("data/new_words.json"))
tid2new = { v: k for k, v in new2tid.items() }
allowed_words = set( new2tid.keys() )
allowed_words_re = re.compile(f'({"|".join(allowed_words)})', flags = re.MULTILINE)

def tknz_encode(x, tokenizer):
    x = ViTokenizer.tknz(x, allowed_words = allowed_words)
    splits = re.split(allowed_words_re, x)
    # print(splits); input() # DEBUG

    token_ids = []
    for i, s in enumerate(splits):
        if i % 2 == 1:
            assert s in allowed_words
            token_ids.append(new2tid[s])
        elif len(s) > 0:
            tids = tokenizer.encode(s, add_special_tokens=False)
            tids = [ old2new_tid(x, tokenizer) for x in tids ]
            token_ids += tids

    token_ids = [ x for x in token_ids if x is not None ]
    return token_ids

def tknz_decode(tids, tokenizer):
    s = ""
    for x in tids:
        if x in tid2new:
            s += tid2new[x]
        else:
            x = new2old[x]
            s += tokenizer.decode(x)
    return s

olds = "Việt Nam thời gian  còn rất dài thực hiện"
tids = tknz_encode(olds, tokenizer)
news = tknz_decode(tids, tokenizer)
assert news.replace("▁", " ") == olds
# print(news)

if __name__ == "__main__":
    import math

    n = len(kept_tids)
    nn = math.ceil(n / 64) * 64

    nnn = n # cách tính khác
    if nnn % 64 != 0:
        nnn += 64 - (nnn % 64)
    assert nn == nnn

    print("kept_tids", n)
    print(n, nn) # 101011 => 101056 (làm tròn để chia hết cho 64)

    print(olds)
    print(news)

    print("tokenizer.vocab_size", tokenizer.vocab_size)
