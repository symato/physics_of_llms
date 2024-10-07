import os, sys, glob, json
from utils_lang import *
from transformers import AutoTokenizer

def get_kept_tids():
    # Keep all special tokens
    kept_tids = set( x for x in range(151643, 151664 + 1) )

    # '''
    from config import ONLINE_MODEL_PATH as model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

def old2new_tid(x, tokenizer):
    if x not in old2new:

        token = tokenizer.decode(x)
        if contains_unwanted(token):
            return None

        for c in "ŀ":
            if c in token:
                return None

        # —including
        

        msg = f">>> old2new_tid error: id {x}, token '{token}'"
        print(msg)
        assert False, msg

    return old2new[x]


if __name__ == "__main__":

    n = len(kept_tids)
    nn = round(n / 64) * 64

    print("kept_tids", n)
    print(n, nn) # 76138 => 76160 (làm tròn để chia hết cho 64)
