import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from utils import *
from transformers import AutoTokenizer
from threading import Thread
import re

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


model_path = "Qwen/Qwen2.5-14B-Instruct"

PATH = f"data/{model_path}"
mkdirs(PATH)


'''
The 4E00—9FFF range covers CJK Unified Ideographs (CJK=Chinese, Japanese and Korean). 
There are a number of lower ranges that relate, to some degree, to CJK:

31C0—31EF CJK Strokes
31F0—31FF Katakana Phonetic Extensions
3200—32FF Enclosed CJK Letters and Months
3300—33FF CJK Compatibility
3400—4DBF CJK Unified Ideographs Extension A
4DC0—4DFF Yijing Hexagram Symbols
4E00—9FFF CJK Unified Ideographs 
'''
min_cjk = ord('\u4e00')
max_cjk = ord('\u9fff')
removed = []

###
def contains_cjk(token):
    for char in token:
        o = ord(char)
        if min_cjk <= o and o <= max_cjk:
            return True
    return False

###
def not_latin(token):
    for char in token:
        if ord(char) > 255:
            return True
    return False

###
def ok(x):
    tid, count = x

    if count < min_count:
        removed.append(x)
        return False

    if count < max_count:
        token = tokenizer.decode(int(tid))
        # Loại nếu không phải chữ latin
        if not_latin(token):
            removed.append(x)
            return False

    return True



tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length = 1024 * 1024 * 4, # 4m ctxlen có thể chứa 1 cuốn sách
)


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


    # Loại bỏ cjk and not_latin
    for k, v in list(count.items()):
        try: token = tokenizer.decode(int(k))
        except: token = None

        if token:
            if contains_cjk(token):
                count.pop(k)
            elif v < 10 and not_latin(token):
                count.pop(k)

    if "last_line_idx" in count:
        count.pop("last_line_idx")

    return count



def get_final_count(input_files):
    if input_files == "stats_mode":
        input_files = glob.glob(f"{PATH}/*_count.json.xz")
        input_files = [ x.replace("_count.json.xz", "") for x in input_files ]
        print(input_files)

    with Pool( processes = num_procs() ) as pool:
        for _ in pool.imap_unordered(get_uniq_tokens, input_files):
            pass

    count = {}
    for infile in input_files:

        x = get_uniq_tokens(infile)
        merge_count(count, x)

    return count


count = get_final_count(input_files)

tid_count_pairs = [ [k, v] for k, v in count.items() ]
total = len(tid_count_pairs)

tid_count_pairs.sort( key = lambda x: -x[1] )

tid_count_pairs = [ x for x in tid_count_pairs if ok(x) ]

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

print(f"{len(tid_count_pairs)} / {total}")


'''
python3 tokens_count.py 3500 20000
87352 / 148986

75608      ">Create"                          3527
31588      "WebElement"                       3527
90835      "invalidate"                       3526
73046      "cych"                             3526
86107      " erklä"                           3526
97599      " dicts"                           3525
69317      "_JOIN"                            3525
86828      "(geometry"                        3524
70076      " MainPage"                        3523
91795      "departments"                      3523
83245      " imdb"                            3523
95435      ".checkbox"                        3523
80325      " DTO"                             3522
97781      "PermissionsResult"                3522
79924      "\titer"                           3522
38913      "_WORLD"                           3522
49931      " TIMER"                           3522
53601      "HasMaxLength"                     3522
85663      ".purchase"                        3521
64668      ".Getter"                          3521
96417      ".playlist"                        3520
31923      ">\";"                             3520
72631      ":hidden"                          3519
66323      "(rgb"                             3519
81374      ".bz"                              3519
91192      " egret"                           3519
84636      ")\":"                             3519
93277      " kuk"                             3518
69462      " Spiele"                          3518
83486      ".answers"                         3518
51439      "getX"                             3517
68915      "(seg"                             3517
55612      "signin"                           3517
96268      "\tcomment"                        3517
62444      " getHeight"                       3516
93354      " WebClient"                       3516
88056      "_residual"                        3516
89459      "(dateTime"                        3516
85464      "[mask"                            3516
8986       " occas"                           3515
95383      "/Base"                            3515
35130      "entai"                            3515
93544      "(SS"                              3515
95267      "(thing"                           3514
78330      "[from"                            3514
71823      "_MARGIN"                          3514
18839      "mercial"                          3514
85524      "_Location"                        3514
53230      "\tmp"                             3514
89875      ".listFiles"                       3513
47519      "THREAD"                           3513
45528      " incarcer"                        3513
86352      " MDB"                             3513
60156      "(speed"                           3512
65641      " ClassNotFoundException"          3512
55575      "_NATIVE"                          3512
62421      "_initialized"                     3511
88633      "isObject"                         3511
80308      "_Stop"                            3510
75139      ".nanoTime"                        3510
75903      "_sentences"                       3510
38285      "adx"                              3510
81422      "BACKGROUND"                       3509
59845      ")});\n"                           3509
80551      "Instantiate"                      3509
57560      "_LICENSE"                         3509
70747      "RenderTarget"                     3509
48885      " })("                             3508
27876      " **)"                             3508
76669      "(priority"                        3508
63827      ".authService"                     3508
73475      "_cred"                            3508
98344      " Wohnung"                         3508
68700      "$I"                               3507
66631      " SIMD"                            3506
81226      "_shot"                            3506
83858      "_ISO"                             3506
31234      "ategori"                          3505
54722      ">}'"                              3505
78085      "\tstat"                           3505
143250     " científico"                      3505
72652      "=settings"                        3505
52508      "%</"                              3505
81497      ".correct"                         3504
74018      "_SEQUENCE"                        3504
48671      " ImageIcon"                       3503
62930      "\tsettings"                       3503
91458      " IEntity"                         3502
80132      " readdir"                         3502
90164      " dct"                             3502
49777      " |\r\n"                           3501
92334      "ULLET"                            3501
41513      "_stride"                          3501
32459      "lld"                              3501
90781      " --------------------"            3501
92938      "PageIndex"                        3501
76993      "_algo"                            3501
69802      ".TextUtils"                       3500
78879      " Npgsql"                          3500
82034      "(ra"                              3500

'''