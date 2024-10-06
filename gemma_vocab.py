import os, sys, glob, json, math

def get_kept_tids():
    kept_tids = set( x for x in range(0, 256) )

    kept_filenames = glob.glob("gemma__1000__20000/tokens_kept__*.jsonl")

    for filename in kept_filenames:
        for line in open(filename, "rt"):
            token, tid, count = json.loads(line)
            kept_tids.add(tid)

    kept_tids = list(kept_tids)
    kept_tids.sort()

    print("new_gemma_vocab", len(kept_tids))
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
        print(">>> old2new_tid error:", x)
        print(tokenizer.decode(x))
        assert False
    else:
        return old2new[x]

if __name__ == "__main__":

    n = len(kept_tids)
    nn = math.ceil(n / 64) * 64

    print("kept_tids", n)
    print(n, nn) # 76138 => 76160 (làm tròn để chia hết cho 64)