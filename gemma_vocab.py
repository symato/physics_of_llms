import os, sys, glob, json, math
from utils_lang import *
from transformers import AutoTokenizer


def get_kept_tids():
    # Keep all special tokens
    kept_tids = set( x for x in range(0, 217) )
    for x in range(255968, 255999 + 1): kept_tids.add(x)

    ''' Gemma vocab khá dị (unigram?)
      "▁EXPERIMENT S",
      "▁EXPERI MENTS",
      "▁EXPER IMENTS",
      "▁negotiator s",
      "▁negoti ators",
      "▁Recharge able",
      "▁Re chargeable",
      "▁espiritual es",
      "▁Multi cultural",
      "▁Mul ticultural",
      "▁potential ities",
    '''
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


STRANGE_TOKENS = set()

def old2new_tid(x, tokenizer):
    global STRANGE_TOKENS

    if x in old2new:
        return old2new[x]

    else:
        token = tokenizer.decode([x])
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


if __name__ == "__main__":

    n = len(kept_tids)
    nn = math.ceil(n / 64) * 64

    print("kept_tids", n)
    print(n, nn) # 76138 => 76160 (làm tròn để chia hết cho 64)


    filename = "../gemma-2-2b-it/tokenizer.model"
    new_filename = "../gemma-2-2b-it__trimm_vocab/tokenizer.model"
    '''
    ## Remove pieces from sentencepiece tokenizer
    # https://huggingface.co/nbroad/donut-base-ascii/blob/main/remove-donut-tokens.ipynb
    from transformers.convert_slow_tokenizer import import_protobuf

    model_pb2 = import_protobuf()
    m = model_pb2.ModelProto()

    m.ParseFromString(open(filename, 'rb').read())
    print(len(m.pieces))
    print(m.pieces[:3])

    print("get kept pieces ...")
    kept_pieces = set([m.pieces[x].piece for x in kept_tids])
    print("kept_pieces", len(kept_pieces))

    print("Remove unkept pieces ...")
    i = 0
    while i < len(m.pieces):

        if m.pieces[i].piece not in kept_pieces:
            m.pieces.pop(i)
        else:
            i += 1
            # print(i)

    with open(new_filename, 'wb') as f:
        f.write(m.SerializeToString())
    # '''

    tokenizer = AutoTokenizer.from_pretrained(filename.split("tokenizer")[0])
    new_tokenizer = AutoTokenizer.from_pretrained(new_filename.split("tokenizer")[0])

    s = "bạn tên là gì? bạn có thể cho tôi biết bạn là ai không"
    s = "số tuổi của An trừ đi số tuổi của Lan là 3, An 10 tuổi hỏi Lan mấy tuổi?"
    tids = tokenizer.encode(s)
    new_tids = new_tokenizer.encode(s)

    print(tids)
    print([tokenizer.decode(x) for x in tids])

    print(new_tids)
    print([new_tokenizer.decode(x) for x in new_tids])

    mapped_tids = [ old2new[x] for x in tids ]
    print(mapped_tids)

    '''
    old_tknz [2, 17534, 2134]
    new_tknz [2, 321, 318, 325, 325, 328, 443, 367, 346, 336, 328, 331, 325, 317]
    mapping  [2, 16149, 2075]

    => new_tknz sau khi remove pieces bị loạn :(
    '''
