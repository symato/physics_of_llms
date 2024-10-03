import os, sys, lzma, glob, json
from multiprocessing import Pool
import re, subprocess

from pyvi import ViTokenizer # pip install pyvi
import fasttext # pip install fasttext
from utils import *

x = ViTokenizer.tokenize("Trường đại học bách khoa hà nội")
x = re.findall(r'[_\w]+', x)
print(x)#; input() # DEBUG

min_count = 0
try:
    # bỏ / ở cuối tham số đầu vào
    x = re.sub(r'/*$', "", sys.argv[1].strip())

    if re.match(r"\d+", x):
        input_files = "stats_mode"
        min_count = int(x)
    else:
        input_files = \
            glob.glob(f"{x}/*.lzma") + \
            glob.glob(f"{x}/*.xz")

except:
    input_files = ["data/test.jsonl.xz"]

print(input_files, min_count)


PATH = f"data/vi_words"
subprocess.run(f"mkdir -p {PATH}", shell = True)


## Fasttext detect lang
langid_model = 'lid.176.bin'
if not os.path.exists('data/' + langid_model):
    cmd = f"wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/{langid_model}; mv {langid_model} data"
    subprocess.run(cmd, shell=True)
FASTTEXT_MODEL = fasttext.load_model('data/' + langid_model)

def detect_lang(text):
    word_re = r'\w+\s'
    words = re.findall(word_re, text)
    words = " ".join(words)
    # Chỉ kiểm tra tiếng Việt với những từ được phân tách rõ ràng

    rs = FASTTEXT_MODEL.f.predict(words, 1, 0.0, 'strict')
    ret = None
    if rs: 
        ret = rs[0][-1].split('__')[-1]
    # print(words, ret); input() # DEBUG
    return ret
assert detect_lang("hello vietnam") == "en"
assert detect_lang( "chào nước mỹ 123sfd http://adf4| tôi là ") == "vi"


def count_words(texts):
    count = {}
    for text in texts:
        if detect_lang(text) == "vi":
            x = ViTokenizer.tokenize(text)
            for word in re.findall(r'[_\w]+', x):

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

            # 1k samples ghi lại kết quả đếm 1 lần
            if idx % 1000 == 999:
                merge_count(count, count_words(texts))
                count["last_line_idx"] = idx

                with lzma.open(outfile, "wt") as f:
                    f.write(json.dumps(count))

                print(f'get_uniq_token {infile}:{count["last_line_idx"]} ...', flush = True)
                texts = []

        # Lần đếm cuối cùng cho chỗ text còn lại
        merge_count(count, count_words(texts))
        count.pop("last_line_idx") # DONE, ko cần ghi lại last_line_idx nữa

        # Ghi kết quả cuối cùng ra file
        with lzma.open(outfile, "wt") as f:
            f.write(json.dumps(count))

        print(f'get_uniq_token {infile} DONE.', flush = True)

    if "last_line_idx" in count:
        count.pop("last_line_idx")

    for w, c in list( count.items() ):
        if "_" not in w or c < min_count:
            count.pop(w)

    return count


def get_final_count(input_files):
    if input_files == "stats_mode":
        input_files = glob.glob(f"{PATH}/*_count.json.xz")
        input_files = [ x.replace("_count.json.xz", "") for x in input_files ]

    count = {}
    with Pool( processes = num_procs() ) as pool:
        for x in pool.imap_unordered(get_uniq_words, input_files):
            merge_count(count, x)

    return count


print("get_final_count ...")
count = get_final_count(input_files)
# print(count)

word_count_pairs = list( count.items() )

word_count_pairs.sort(key = lambda x: -x[1]) # sắp xếp giảm dần theo count

maxx = 25
spaces = " " * 100

with open("data/vi_words_count.txt", "wt") as f:
    for w, c in word_count_pairs:
        if "_" in w:
            f.write(f"{w}{spaces[maxx - len(w)]} {c}\n")


'''

python3 vi_words_count.py 1000

wc -l data/vi_words_count.txt

tail data/vi_words_count.txt

'''