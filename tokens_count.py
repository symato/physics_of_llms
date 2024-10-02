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


remains = set(wanted_tids)
for tid, _ in removed:
    tid = int(tid)
    if tid in remains:
        remains.remove(tid)

print(f"{len(remains)} / {tokenizer.vocab_size}")

with open("tokens_removed.txt", "wt") as f:
    for tid, count in removed:
        f.write(pretty(tid, count) + "\n")

'''

python3 tokens_count.py 380
wanted 105377 / 151643
stats_mode
380 0
mkdir -p data
mkdir -p data/Qwen
mkdir -p data/Qwen/Qwen2.5-14B-Instruct
get_final_count ...
remove_not_ok_pairs ...
sort remain_pairs and removed ...

=== Một số removed tokens ===

65756      "\tIterator"                        379
79408      "CLLocation"                        379
83172      "ADX"                               379
67254      " Produto"                          379
37064      " �"                                379
79086      "praak"                             379
92444      " ActivityCompat"                   379
79951      "_sta"                              379
54642      "�"                                 379
75782      ".parseFloat"                       379
77107      "_DET"                              379
86428      "']*"                               379
68082      " AutoMapper"                       379
98853      "(rowIndex"                         379
46269      "$name"                             379
76093      " ise"                              379
50897      ".ibatis"                           379
87586      "']))\n\n"                          379
45705      "(\"\")]\n"                         379
38001      "VERRIDE"                           379
45521      " }}>"                              379
91703      "_UPPER"                            379
99264      "�"                                 379
71332      " DRV"                              379
96619      "_pedido"                           379
72361      ":host"                             379
147668     "ŵ"                                 379
54013      "ESSAGES"                           379
81071      "perfil"                            378
57073      "!\");\n\n"                         378
85735      "typings"                           378
131519     " tầ"                               378
91868      "(accounts"                         378
90350      ";left"                             378
83163      " onPostExecute"                    378
89259      "@Bean"                             378
59831      "_WHITE"                            378
76689      "(vehicle"                          378
66936      "_DEN"                              378
93104      "_ped"                              378
61008      " Qgs"                              378
56768      " bcm"                              378
68388      "_WS"                               378
71436      " BrowserAnimationsModule"                                                                                                                  378
97673      "_COMPANY"                          377
90066      "ıkl"                               377
86458      " McCart"                           377
57384      "(MSG"                              377
91896      " schöne"                           377
68474      " FirebaseDatabase"                 377
40503      "_DEFIN"                            377
92787      "_DRIVE"                            377
51387      " },{\n"                            377
92700      "[char"                             377
79443      " erót"                             377
88265      ".toolbox"                          377
96421      "\tbl"                              377
84193      " selber"                           377
30409      "__(("                              377
27042      " *);\n"                            377
49310      "setPosition"                       377
92837      "InputDialog"                       377
64026      "_Timer"                            377
56817      "removeAttr"                        377
88674      "_DISPATCH"                         377
48697      " svens"                            376
89607      "_WRONG"                            376
68555      " DataAccess"                       376
96205      " equipe"                           376
87567      "//------------------------------------------------------------------------------\n\n"                                                      376
77348      "_USED"                             376
81175      "textAlign"                         376
52016      "[contains"                         376
45659      ".userInfo"                         376
62374      "TexParameteri"                     376
75688      "_UNICODE"                          376
56206      " modificar"                        376
98302      "(guess"                            376
90132      "_LOWER"                            376
20979      "******/"                           376
71212      ".Roll"                             376
97360      "_ATOM"                             376
87843      "_ATOMIC"                           376
36417      "_PAY"                              375
74912      "\tmask"                            375
60902      "UGC"                               375
77299      "_MIX"                              375
87183      "toolbox"                           375
77634      " freund"                           375
59315      "_userdata"                         375
133893     " milhões"                          375
97342      "(dw"                               375
87260      " \".\");\n"                        375
76605      "\tExt"                             375
99073      ".JWT"                              375
51873      " treffen"                          375
97214      "InnerHTML"                         375
135321     " própria"                          375
90323      "autoplay"                          375
63552      ",email"                            375


81063      "edith"                             165
59747      " CGPointMake"                      165
145626     "➠"                                 165
77005      ".dateTimePicker"                   165
96463      "\\Catalog"                         165
96285      "pesan"                             165
148306     "ḏ"                                 165
62005      "�"                                 165
61707      "¯¯¯¯"                              165
27367      " helicopt"                         164
8032       "�"                                 164
148595     "ℜ"                                 164
49348      ":^{\n"                             164
81956      "))];\n"                            164
88841      "mensagem"                          164
34811      "��"                                164
56980      "+=("                               164
32109      " antibiot"                         164
147196     "Į"                                 164
126763     "ladı"                              164
35668      "\">'.$"                            164
77855      "SelfPermission"                    164
98196      "UIImagePickerController"           164
90552      " pinMode"                          164
42303      "(__('"                             164
85447      " \"\"\"\",\n"                      164
64579      "érience"                           164
66994      "_MUTEX"                            164
41800      " }}\">"                            164
83795      "\t\t\t\t\t\t\t\t\t\t "             164
40599      " millenn"                          164
84839      " sağ"                              163
129947     " üzerinde"                         163
95005      "SectionsIn"                        163
145253     "▰"                                 163
58868      "kontakte"                          163
62661      "$array"                            163
73162      " ()=>{\n"                          163
88650      "\tmkdir"                           163
93795      "：%"                               163
98817      "_Panel"                            163
65035      "ecedor"                            163
96980      "[$_"                               163
95345      "\"]=="                             163
96359      "@PostMapping"                      163
128659     "łam"                               163
127860     "ędzi"                              163
99220      "�"                                 163
99100      "idUser"                            163
61810      "uvw"                               163
95047      "+\"<"                              163
97877      " conexao"                          163
93699      " gridColumn"                       163
90483      " QTableWidgetItem"                 163
83367      " nilai"                            163
86695      "categorie"                         163
41226      "VERTISEMENT"                       162
63485      "(HWND"                             162
97236      ">');"                              162
53995      " ''){\n"                           162
91751      " SpringApplication"                162
60868      " onChangeText"                     162
145584     "∠"                                 162
45256      "\\HttpFoundation"                  162
94335      " '\"';\n"                          162
76150      " didFinish"                        162
73062      "/*------------------------------------------------"                                                                                        162
84252      "_VC"                               162
91051      "Tambah"                            162
61851      "$field"                            162
136800     " Gerät"                            162
50459      "_PWM"                              162
72002      "(Gtk"                              162
59544      ")localObject"                      162
35146      "��"                                162
145327     "𝑜"                                 162
73445      "DevExpress"                        162
24904      ";?>"                               162
37913      "�"                                 162
49905      ".XtraPrinting"                     162
82672      " }}\r\n"                           162
58007      "/****************************************************************************\n"                                                           161
140299     "gänge"                             161
21885      "\\Eloquent"                        161
87103      ".contentOffset"                    161
98751      "AdminController"                   161
55811      "<count"                            161
35361      ".XtraGrid"                         161
146992     "Ő"                                 161
61573      "ImplOptions"                       161
85289      " setFrame"                         161
83849      "�"                                 161
123867     "�"                                 161
129245     " açık"                             161
41951      " damer"                            161
95365      "Periph"                            161
76268      "contenido"                         161
65049      " QRect"                            161
66108      " GtkWidget"                        161
93424      "bersome"                           161


71195      " sexle"                              3
89637      "HomeAs"                              3
89832      " analsex"                            3
96835      " sextreffen"                         3
142468     " porówna"                            3
23394      "SequentialGroup"                     3
138968     " móg"                                3
70237      "+lsi"                                3
141399     " yönetici"                           3
143555     " müdah"                              3
76035      " datingside"                         3
47357      " pornô"                              3
143547     " mük"                                3
81712      " beurette"                           3
95585      "]=]"                                 3
74887      " odense"                             3
143839     " yayg"                               2
56669      "LIBINT"                              2
139210     " ülkem"                              2
84047      "Ubergraph"                           2
91239      "<UFunction"                          2
90196      "CppI"                                2
142092     " pobli"                              2
76889      " \\<^"                               2
143452     " tecrübe"                            2
133118     " mücade"                             2
86289      "\tUObject"                           2
137728     "légi"                                2
127545     " düzenle"                            2
140873     " jednocze"                           2
83804      "gMaps"                               2
37735      "_StaticFields"                       2
39957      " bakeca"                             2
143855     " vazge"                              2
93552      "BracketAccess"                       2
143447     " proprié"                            2
14278      " /*<<<"                              2
139530     " günl"                               2
61902      " sexdate"                            2
85791      " sidl"                               2
97971      "-cmpr"                               2
128171     "lararas"                             2
72128      "_InternalArray"                      2
140324     " düzenlenen"                         2
138173     "Cumhur"                              2
142582     " bölgesinde"                         2
98068      "SmartyHeaderCode"                    2
84369      "/tinyos"                             2
70270      " pornofil"                           2
24094      ":UIControl"                          2
78508      "[MAXN"                               2
52209      "arsimp"                              1
44046      "%timeout"                            1
140071     " gündem"                             1
138155     " seçen"                              1
136454     " sürek"                              1
143783     " uçu"                                1
56622      " ;;="                                1
90297      " [=["                                1
74161      "lbrakk"                              1
74084      "rbrakk"                              1
88887      "lparr"                               1
88920      "rparr"                               1
50245      ")paren"                              1
133697     " Üniversites"                        1
139890     " sürecin"                            1
70290      " PodsDummy"                          1
143838     " vücud"                              1
39170      "wcsstore"                            1
64792      " vivastreet"                         1
49511      "VMLINUX"                             1
137583     "géni"                                1
31283      " neuken"                             1
137548     "fício"                               1
84962      "GameObjectWithTag"                   1
71507      "DECREF"                              1
23086      " [-]:"                               1
58739      "ConstraintMaker"                     1
81368      " uLocal"                             1
62685      " RTWF"                               1
56319      "_Statics"                            1
70266      "drFc"                                1
96481      " JSName"                             1
44694      ">tagger"                             1
91154      " XPAR"                               1
23543      "<lemma"                              1
78042      "\">';\r\n"                           1
52646      "yyval"                               1
65509      "_hresult"                            1
126390     "maktad"                              1
89234      " pornofilm"                          1
128133     "prowadzi"                            1
143269     "pósito"                              1
86278      " sexkontakte"                        1
132815     " mümk"                               1
139914     " Müdürü"                             1
78593      "_:*"                                 1
71918      " StreamLazy"                         1
24962      "methodVisitor"                       1
98372      " *}\n\n"                             1


95625 / 151643
'''
