import os, sys, lzma, glob, json
from multiprocessing import Pool
from functools import partial
from utils import *
from transformers import AutoTokenizer
from threading import Thread
import re

model_path = "Qwen/Qwen2.5-14B-Instruct"
model_path = "meta-llama/Llama-3.1-70B-Instruct"

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
        return False

    token = tokenizer.decode(int(tid))

    if count >= max_count:
        # Loại nếu chứa cjk
        if contains_cjk(token):
            return False            
    else:
        # Loại nếu không phải chữ latin
        if not_latin(token):
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


    if "last_line_idx" in count:
        count.pop("last_line_idx")

    # Loại bỏ cjk and not_latin
    for k, v in list(count.items()):
        token = tokenizer.decode(int(k))

        if contains_cjk(token):
            count.pop(k)

        elif v < 10 and not_latin(token):
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

137966     " нем"                            19977
77334      "します"                          19945
142055     " Ю"                              19922
132366     "ением"                           19906
175        "�"                               19888
124639     "そう"                            19884
95106      "’)"                              19882
125201     "раж"                             19829
126604     "инг"                             19745
126811     "нь"                              19737
129297     " кар"                            19735
130290     "ному"                            19701
25715      "자"                              19687
126711     " дом"                            19684
128798     " того"                           19683
127131     "Ш"                               19632
55959      "азв"                             19630
39511      "ź"                               19617
135226     "ダン"                            19599
129162     " где"                            19595
73154      " дв"                             19582
53586      "ации"                            19575
127082     "един"                            19527
134227     " пок"                            19509
61676      "ления"                           19489
127015     "ово"                             19482
144523     "➡"                               19455
132339     "ело"                             19343
125308     "рен"                             19315
98946      "де"                              19296
20879      "�"                               19293
125223     "це"                              19284
129244     "руж"                             19276
58899      "�"                               19272
9722       "�"                               19259
145925     "Ỵ"                               19258
90777      " уже"                            19254
75525      "ב"                               19148
129917     " Он"                             19142
151276     "�"                               19139
50230      "この"                            19111
125191     " гор"                            19084
58411      "ש"                               19074
95385      "оля"                             19042
144129     "〉"                              19014
52798      "�"                               19004
127011     "ブル"                            18997
126228     "ме"                              18991
146036     "❉"                               18980
124797     "ду"                              18978
126394     "бо"                              18976
124662     "си"                              18917
125626     "вар"                             18894
125258     "ден"                             18882
30785      "ิ"                               18867
125286     "рас"                             18837
145082     "ň"                               18799
45577      " ف"                              18759
39697      "ز"                               18739
28319      "ี"                               18677
128992     " ее"                             18674
144867     "♠"                               18657
148414     "ರ"                               18626
94304      " ≠"                              18571
126362     "ней"                             18570
99012      "ி�"                              18564
127126     "цев"                             18494
128856     " том"                            18484
41312      "া"                               18470
21460      "██"                              18430
96155      " �"                              18428
146770     "ܫ"                               18414
144185     "😂"                              18411
20184      "น"                               18407
84764      "ости"                            18403
125046     "モ"                              18378
144781     "˚"                               18356
141688     " пон"                            18356
16116      "ın"                              18350
147952     "ܩ"                               18339
72661      "олов"                            18319
128647     "са"                              18290
84487      "На"                              18275
43871      "レ"                              18267
133049     " поп"                            18256
128729     " они"                            18252
70354      "ţi"                              18250
130201     "ению"                            18218
125717     " пл"                             18214
124972     " соб"                            18198
56226      "プ"                              18189
145454     "ђ"                               18189
133962     " рек"                            18181
125608     "лас"                             18175
44993      "….\n\n"                          18167
97751      " который"                        18166
19841      "้"                               18158
126252     "се"                              18142
98967      "…it"                             18134
47985      "도"                              18121


143835     " 넘"                               397
83777      "\"]=$"                             397
142603     "プリン"                            397
141406     "しておく"                                  397
142170     "スペース"                                  397
143749     "езжа"                              397
128655     "جو"                                397
129132     "نو"                                397
130918     " الرياض"                           397
133808     "เกษ"                               397
134685     " chcia"                            396
144361     "ѣ"                                 396
128522     " حت"                               396
141        "�"                                 396
133487     " 어느"                             396
133329     " 이해"                             396
136164     "عامل"                              396
126615     " 그런"                             396
129485     " özel"                             395
72629      " pupper"                           395
125210     "วด"                                395
126750     "ówi"                               395
65024      "NoArgsConstructor"                 395
23574      "CppMethod"                         395
127491     "ąda"                               395
148281     "ʧ"                                 395
122448     "�"                                 395
135231     "すこと"                            395
146472     "ۇ"                                 394
73745      "(strtolower"                       394
137178     " Yönet"                            394
146634     "◢"                                 394
125503     " العمل"                            394
134693     " çıkt"                             394
128034     " yı"                               394
70920      " voksen"                           394
129781     " الناس"                            394
143862     " 느"                               394
134920     "intérieur"                         394
129767     "بك"                                394
126376     "רש"                                394
125354     "הפ"                                394
45216      "@endif"                            393
73466      "\":@\""                            393
130457     "tör"                               393
127245     "malı"                              393
90838      ";]/"                               392
65019      "(IDC"                              392
22210      " erotische"                        392
74800      " QLatin"                           392
146849     "ᴥ"                                 392
124643     "רים"                               392
84999      "/******************************************************************************/\n"                                                        392
20047      "�"                                 392
10738      "CLUD"                              391
94563      "\\Helpers"                         391
61660      "-'.$"                              391
128848     " önemli"                           391
131681     "dzić"                              391
141228     " jakieś"                           391
127247     "егодня"                            391
130412     "dım"                               391
125309     "اسي"                               391
95575      "_CNTL"                             390
142241     " nadzie"                           390
142776     "뷰"                                390
124295     "ปล"                                390
151462     "ረ"                                 390
124285     "ــ"                                390
124359     "قو"                                390
139627     "좌"                                390
141355     "естеств"                           390
125332     " البر"                             390
145911     "ѐ"                                 390
145928     "ㅏ"                                389
125178     "ישראל"                             389
137071     " kaldır"                           389
90547      "$insert"                           389
29823      "ernetes"                           389
124625     "สามารถ"                            389
135488     "をご"                              389
126885     "ทาน"                               389
127956     " olmad"                            388
91947      "\\Redirect"                        388
96610      "KANJI"                             388
80575      "\"struct"                          388
145429     "👀"                                388
131572     "ないので"                                  388
146249     "🌎"                                388
144063     "콜"                                388
97991      "OPTARG"                            388
126456     "فن"                                388
128374     " خلال"                             388
147751     "Ꮒ"                                 388
147391     "𝐿"                                 388
138499     " предоставля"                      387
51117      "\tstrcat"                          387
22719      "ICollectionView"                   387
128860     "stęp"                              387
125096     "צר"                                387


137973     "に�"                                10
135920     " الفت"                              10
135928     " לומר"                              10
141867     " włosów"                            10
134268     "سؤ"                                 10
141148     "أور"                                10
149079     "꼍"                                 10
126390     "maktad"                              9
143555     " müdah"                              9
143855     " vazge"                              9
85749      "\tRuntimeObject"                     9
52209      "arsimp"                              9
27530      ">\\<^"                               9
141847     " rahats"                             9
70523      " técn"                               9
84865      " datingsider"                        9
140071     " gündem"                             8
88887      "lparr"                               8
76889      " \\<^"                               8
143783     " uçu"                                8
90745      " pornôs"                             8
130670     "bilità"                              8
93552      "BracketAccess"                       8
54616      " ?>\r\n\r\n"                         8
98018      "-Cds"                                8
139705     " dünyan"                             8
90196      "CppI"                                8
78593      "_:*"                                 8
81368      " uLocal"                             7
70237      "+lsi"                                7
53178      "CppGeneric"                          7
74379      "$LANG"                               7
141514     " çerç"                               7
23543      "<lemma"                              7
55125      ".sulake"                             7
139225     "sistência"                           6
70290      " PodsDummy"                          6
75271      "LANGADM"                             6
30727      "{EIF"                                6
97971      "-cmpr"                               6
34822      "CppMethodPointer"                    6
81943      " bakeka"                             6
135685     " któr"                               6
138155     " seçen"                              6
88920      "rparr"                               5
70266      "drFc"                                5
71916      " EnumerableStream"                   5
96636      "\tRTCT"                              5
58490      "IntoConstraints"                     5
143548     " mükem"                              5
51262      "SpecWarn"                            5
141803     " désorm"                             4
84047      "Ubergraph"                           4
57966      " hexatrigesimal"                     4
71645      " XBOOLE"                             4
137548     "fício"                               4
56261      "_REALTYPE"                           4
49511      "VMLINUX"                             4
88343      " kutje"                              3
67089      "_UClass"                             3
72128      "_InternalArray"                      3
70127      " FINSEQ"                             3
42487      "atrigesimal"                         3
40359      "rigesimal"                           3
67796      ";\r\r\r\n"                           3
68861      "CHKERRQ"                             3
45854      " swingerclub"                        3
74420      " NUITKA"                             3
131653     " birç"                               2
142818     " zobow"                              2
139042     " dóla"                               2
62685      " RTWF"                               2
86278      " sexkontakte"                        2
49074      "PlainOldData"                        2
89312      "selectorMethod"                      2
139890     " sürecin"                            2
60863      "_RGCTX"                              2
61488      "rgctx"                               2
71568      "'])){\r\n"                           2
44046      "%timeout"                            2
95637      "departureday"                        2
141414     " Müslü"                              2
86841      "$fdata"                              2
96684      " JSBracketAccess"                    1
44694      ">tagger"                             1
137568     " ücrets"                             1
139570     "powiedzie"                           1
93973      "-vesm"                               1
74472      "aincontri"                           1
128199     "przedsi"                             1
78778      " \"(\\<"                             1
88371      "useRal"                              1
49225      "adaptiveStyles"                      1
86923      "/ayushman"                           1
139034     "nquête"                              1
54714      " ZeroConstructor"                    1
53623      " IsPlainOldData"                     1
136454     " sürek"                              1
78640      " Hexatrigesimal"                     1
83576      " wannonce"                           1

'''