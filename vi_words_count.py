import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess
import transformers

from mypyvi import ViTokenizer
from utils import *
from utils_lang import *

x = ViTokenizer.tknz("Trường đại học bách khoa hà nội")
x = re.findall(r'[▁\w]+', x)
print(x)#; input() # DEBUG

try: x = sys.argv[1]
except: x = "data/test.jsonl.xz"

min_count = 0

# bỏ / ở cuối tham số đầu vào
x = re.sub(r'/*$', "", x.strip())

if re.match(r"\d+", x):
    input_files = "stats_mode"
    min_count = int(x)
else:
    if os.path.isdir(x):
        input_files = glob.glob(f"{x}/*")
    else:
        input_files = [x]

print(input_files, min_count)


PATH = f"data/vi_words"
subprocess.run(f"mkdir -p {PATH}", shell = True)


def count_words(texts):
    count = {}
    for text in texts:
        if True:#detect_lang(text) == "vi":
            x = ViTokenizer.tknz(text)
            
            words_without_space_prefix = re.findall(r'(?<![ ▁\w])[▁\w]+', x)
            words_with_space_prefix    = re.findall(r' [▁\w]+', x)

            for word in words_without_space_prefix + words_with_space_prefix:

                if word not in count:
                    count[word] = 0

                count[word] += 1

    return count


def merge_count(count, x):
    for k, v in x.items():

        if k not in count:
            count[k] = 0

        count[k] += v


def get_uniq_words(infile):
    x = infile.split("/")[-1]
    outfile = f"{PATH}/{x}_count.json"
    print(outfile)


    try:
        if os.path.exists(outfile):
            count = json.load(open(outfile))
        else:
            count = json.load(lzma.open(outfile + ".xz"))
    except:
        count = { "last_line_idx": 0 }

    if os.path.exists(infile) and "last_line_idx" in count: # DONE

        texts = []

        if ".xz" in infile or ".lzma" in infile:
            f = lzma.open(infile)
        else:
            f = open(infile)

        for idx, line in enumerate( f ):
            if idx <= count["last_line_idx"]:
                continue

            data = json.loads(line)
            if "text" in data:
                t = data["text"]
                if isinstance(t, str):
                    texts.append( t )

            if "conversations" in data:
                for c in data["conversations"]:
                    v = c["value"]
                    if isinstance(v, str):
                        texts.append( v )

            # 5k samples ghi lại kết quả đếm 1 lần
            if idx % 5000 == 4999:
                merge_count(count, count_words(texts))
                count["last_line_idx"] = idx

                with open(outfile, "wt") as f:
                    f.write(json.dumps(count, ensure_ascii = False))

                print(f'get_uniq_token {infile}:{count["last_line_idx"]} ...', flush = True)
                texts = []

        # Lần đếm cuối cùng cho chỗ text còn lại
        merge_count(count, count_words(texts))
        count.pop("last_line_idx") # DONE, ko cần ghi lại last_line_idx nữa

        # Ghi kết quả cuối cùng ra file
        with open(outfile, "wt") as f:
            f.write(json.dumps(count, ensure_ascii = False))

        print(f'get_uniq_token {infile} DONE.', flush = True)

        f.close()

    if "last_line_idx" in count:
        count.pop("last_line_idx")

    return count


def get_final_count(input_files):
    if input_files == "stats_mode":
        input_files = glob.glob(f"{PATH}/*_count.json.xz")
        input_files = [ x.replace("_count.json.xz", "") for x in input_files ]

    count = {}
    with Pool( processes = num_procs() ) as pool:
        for x in pool.imap_unordered(get_uniq_words, input_files):
            merge_count(count, x)

    for w, c in list( count.items() ):
        if "▁" not in w or c < min_count:
            count.pop(w)

        elif " " == w[0]:
            count.pop(w)
            w = w[1:]
            if w not in count: count[w] = 0
            count[w] += c

    return count


print("get_final_count ...")
count = get_final_count(input_files)
# print(count)


tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
words = []

for word, freq in count.items():

    text = word.replace("▁", " ")
    tids = tokenizer.encode(text)
    tids_ = tokenizer.encode(" " + text)

    if len(tids_) < len(tids):
        tids = tids_

    qwen_tokens = [ tokenizer.decode(tid) for tid in tids ]
    qwen_tokens_count = len(qwen_tokens)
    impact = freq * (qwen_tokens_count - 1) # trừ đi 1 slot là tid mới thêm vào

    words.append({
        "word": word,
        "impact": impact,
        "qwen_tokens": qwen_tokens, 
        "freq": freq,
        "qwen_tokens_count": qwen_tokens_count,
    })


words.sort(key = lambda x: -x["impact"])

maxx = 35
spaces = " " * (maxx + 1)

with open("data/vi_words_impact.jsonl", "wt") as f:
    for w in words:
        if w["impact"] > 0:
            a, b = json.dumps(w, ensure_ascii = False).split(", ", 1)
            f.write(f"{a},{spaces[:maxx - len(a)]}{b}\n")


'''

python3 vi_words_count.py 6000

head -n 10 data/vi_words_impact.jsonl

wc -l data/vi_words_impact.jsonl

'''