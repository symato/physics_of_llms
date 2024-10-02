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

    token = tokenizer.decode(int(tid))

    if count < min_count:
        if not contains_emoji(token):
            return False

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

# remain_pairs = []
removed = []

print("remove_not_ok_pairs ...")
with Pool( processes = num_procs() ) as pool:
    for keep, remove in pool.imap_unordered(remove_not_ok_pairs, chunks):
        # remain_pairs += keep
        removed += remove

print("sort remain_pairs and removed ...")
# remain_pairs.sort( key = lambda x: -x[1] )
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

print("\n= = = Một số removed tokens = = =\n")
for tid, count in x:
    if count == 0:
        print("\n")
        continue

    if tid != "last_line_idx":
        token = json.dumps(tokenizer.decode(int(tid)), ensure_ascii = False)
        n = len(token)
        print(f"{tid}{spaces[:10 - len(tid)]} {token}{spaces[:maxx - n]}\t{count:10.0f}")

remains = tokenizer.vocab_size - len(removed)
print(f"{remains} / {tokenizer.vocab_size}")


'''

python3 tokens_count.py 600
wanted 105377 / 151643
stats_mode
600 0
mkdir -p data
mkdir -p data/Qwen
mkdir -p data/Qwen/Qwen2.5-14B-Instruct
get_final_count ...
remove_not_ok_pairs ...
sort remain_pairs and removed ...
80551      "Instantiate"                       599
56861      ".notifyDataSetChanged"             599
17631      "STANCE"                            599
36087      "(win"                              599
52044      " tylko"                            599
84238      " �"                                599
77127      "(deg"                              599
87451      "SAME"                              599
34996      ";k"                                599
58995      "_ER"                               599
43435      "textContent"                       599
61255      "qry"                               599
93697      " bordel"                           599
28577      " UICollectionView"                 599
77992      ".accel"                            599
83600      " Produk"                           599
78359      "_PADDING"                          599
93020      "Hashtable"                         599
70779      " onTouch"                          598
84220      " oferta"                           598
95551      "WiFi"                              598
61957      "sendMessage"                       598
35434      "@author"                           598
33237      ";\"></"                            598
57325      " Reached"                          598
62444      " getHeight"                        598
97715      "complexContent"                    598
95879      ".getMin"                           598
75637      "_quotes"                           598
34513      "\t                   "             598
53736      "(rv"                               598
92781      "opcion"                            598
135        "�"                                 598
64374      " userList"                         598
99130      ".addNode"                          598
62039      ".FieldName"                        598
72439      " vegas"                            598
33710      "addContainerGap"                   598
60992      "maxLength"                         597
49987      " \n        \n"                     597
76987      ",path"                             597
30740      "-muted"                            597
87166      " Dysfunction"                      597
86477      "Hibernate"                         597
14222      "strcmp"                            597
76993      "_algo"                             597
79655      " scrollbar"                        597
87653      "$\")\n"                            597
90832      "(Handle"                           597
80566      ".fp"                               597
5102       "ablish"                            597
95838      "/array"                            597
36057      "_appro"                            597
75623      ".setWidth"                         597
82906      " setResult"                        597
94199      ".MODEL"                            597
90293      "isify"                             597
79329      ".imageUrl"                         597
96763      "/terms"                            597
31593      "_ASSOC"                            597
48802      " senha"                            597
68544      "_rb"                               597
73186      "_ALLOWED"                          597
88596      "invoices"                          597
97089      "/left"                             596
52228      "\tcomponent"                       596
73805      " ['/"                              596
90707      "<d"                                596
84198      ",no"                               596
32374      "addEventListener"                  596
90052      "-Owned"                            596
66297      "(dl"                               596
87444      ".AddField"                         596
62149      ".BatchNorm"                        596
70625      "_PRIV"                             596
79594      " exemplo"                          596
34032      " Frequ"                            596
96415      "Clazz"                             596
52805      "='_"                               596
81103      " toDate"                           596
7879       " QString"                          596
138918     " Böl"                              596
96030      " qx"                               596
69238      " tamanho"                          596
31670      "\t         "                       596
91203      " MouseButton"                      596
74312      "(fr"                               596
82827      "-bars"                             596
63841      "_BREAK"                            596
60430      "_SCROLL"                           596
45836      "oomla"                             596
73511      " poil"                             596
96548      "_reservation"                      596
69230      "]}\""                              595
54973      "='{"                               595
92108      "()]\n\n"                           595
98666      " eens"                             595
97128      ":border"                           595
57240      "processable"                       595
55495      " dbContext"                        595


145214     "❥"                                 366
79503      "_dw"                               366
33077      "(($"                               366
50345      "mozilla"                           366
48424      "\tBuffer"                          366
51012      "(gameObject"                       366
30999      "']=="                              366
75912      "(cd"                               366
88714      "Contours"                          366
59373      "\tJSONObject"                      366
70352      "\tButton"                          366
86802      "_CYCLE"                            366
141257     " nächste"                          366
72374      "_combine"                          366
88116      "designation"                       366
49403      "GetName"                           366
87168      " jMenuItem"                        366
40503      "_DEFIN"                            366
134189     " déco"                             366
86460      "buie"                              366
47455      " �"                                366
81073      "_fence"                            366
164        "�"                                 366
59315      "_userdata"                         366
95260      "=event"                            366
47523      " Antar"                            366
74179      "_fre"                              366
69754      "uforia"                            365
30509      "anggal"                            365
74082      " BaseEntity"                       365
59303      "/epl"                              365
79841      "[hash"                             365
70235      " '.')"                             365
73552      "AZY"                               365
74528      "[\"@"                              365
77057      "(Have"                             365
99301      "�"                                 365
89318      " ============================================================================\n"                                                           365
91596      ".stamp"                            365
63845      "Titulo"                            365
65668      "       \n\n"                       365
41335      "layui"                             365
71675      " retorna"                          365
95742      "_MODULES"                          365
87843      "_ATOMIC"                           365
60235      "(ele"                              365
58558      "_BTN"                              365
92012      " algumas"                          365
72620      ".getContentPane"                   365
73658      " rowData"                          365
94722      "])**"                              365
145170     "➜"                                 365
74311      ".clientHeight"                     365
97605      "']])\n"                            365
76500      " shm"                              364
96009      "_WATCH"                            364
75782      ".parseFloat"                       364
84731      "getToken"                          364
88286      " UserDetails"                      364
58872      "--;\n\n"                           364
98767      "NavigationItemSelectedListener"                                                                                                            364
56590      "_fifo"                             364
79386      "_SID"                              364
63494      "arendra"                           364
67712      "@ResponseBody"                     364
68554      ".JSONArray"                        364
85516      "_erase"                            364
59274      ".BorderColor"                      364
68330      "viar"                              364
96185      " Filme"                            364
90350      ";left"                             364
76689      "(vehicle"                          364
95980      " dbHelper"                         364
123934     "�"                                 364
94241      " GRAT"                             364
79570      ".onView"                           364
96781      "SearchTree"                        364
94180      "_ORIENTATION"                      364
41998      "(Py"                               364
68388      "_WS"                               364
32127      "-labelledby"                       363
50303      "anyak"                             363
72006      "\">//"                             363
90533      " bpp"                              363
92041      "fcn"                               363
33183      "celed"                             363
4902       " purch"                            363
35115      "*/\n\n\n"                          363
69690      ".ogg"                              363
88744      " ('\\"                             363
35857      " Stateless"                        363
77299      "_MIX"                              363
45153      "++++++++++++++++"                  363
90712      "=wx"                               363
87527      " ksi"                              363
91834      "_registro"                         363
89538      ".Css"                              363
141373     " giấ"                              362
46269      "$name"                             362
81091      ")];"                               362


50279      "<ll"                                 3
92149      "();\r\r\n"                           3
97697      "APolynomial"                         3
87257      "_Printf"                             3
51796      "__(/*!"                              3
89637      "HomeAs"                              3
33857      " +#+#+#+#+#+"                        3
41864      "(EIF"                                3
89832      " analsex"                            3
96835      " sextreffen"                         3
62521      "SetBranch"                           3
43732      " nettsteder"                         3
142468     " porówna"                            3
141399     " yönetici"                           3
143555     " müdah"                              3
71195      " sexle"                              3
84047      "Ubergraph"                           2
91599      "\"urls"                              2
143839     " yayg"                               2
140324     " düzenlenen"                         2
70523      " técn"                               2
139530     " günl"                               2
142092     " pobli"                              2
69236      " spep"                               2
61902      " sexdate"                            2
137728     "légi"                                2
85791      " sidl"                               2
97971      "-cmpr"                               2
128171     "lararas"                             2
72128      "_InternalArray"                      2
143447     " proprié"                            2
138173     "Cumhur"                              2
142582     " bölgesinde"                         2
98068      "SmartyHeaderCode"                    2
143855     " vazge"                              2
93552      "BracketAccess"                       2
84369      "/tinyos"                             2
140873     " jednocze"                           2
14278      " /*<<<"                              2
76889      " \\<^"                               2
143452     " tecrübe"                            2
133118     " mücade"                             2
56669      "LIBINT"                              2
91239      "<UFunction"                          2
86289      "\tUObject"                           2
127545     " düzenle"                            2
90196      "CppI"                                2
39957      " bakeca"                             2
70270      " pornofil"                           2
139210     " ülkem"                              2
78508      "[MAXN"                               2
83804      "gMaps"                               2
37735      "_StaticFields"                       2
136454     " sürek"                              1
143783     " uçu"                                1
86278      " sexkontakte"                        1
132815     " mümk"                               1
44046      "%timeout"                            1
62685      " RTWF"                               1
56319      "_Statics"                            1
130670     "bilità"                              1
23543      "<lemma"                              1
78042      "\">';\r\n"                           1
52646      "yyval"                               1
139914     " Müdürü"                             1
78593      "_:*"                                 1
71918      " StreamLazy"                         1
24962      "methodVisitor"                       1
98372      " *}\n\n"                             1
49511      "VMLINUX"                             1
137583     "géni"                                1
31283      " neuken"                             1
23086      " [-]:"                               1
58739      "ConstraintMaker"                     1
81368      " uLocal"                             1
50245      ")paren"                              1
133697     " Üniversites"                        1
139890     " sürecin"                            1
70290      " PodsDummy"                          1
56622      " ;;="                                1
90297      " [=["                                1
74161      "lbrakk"                              1
74084      "rbrakk"                              1
143838     " vücud"                              1
65509      "_hresult"                            1
126390     "maktad"                              1
89234      " pornofilm"                          1
128133     "prowadzi"                            1
143269     "pósito"                              1
52209      "arsimp"                              1
24094      ":UIControl"                          1
140071     " gündem"                             1
138155     " seçen"                              1
137548     "fício"                               1
84962      "GameObjectWithTag"                   1
71507      "DECREF"                              1
70266      "drFc"                                1
96481      " JSName"                             1
44694      ">tagger"                             1
91154      " XPAR"                               1

86715 / 151643

'''