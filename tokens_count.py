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

    # Ưu tiên mọi token có chứa dù chỉ 1 ký tự đặc trưng tiếng Việt
    if canbe_vietnamese(token):
        return True

    # Nếu chứa những ký tự của ngôn ngữ lạ như cjk, thailand ... loại
    if contains_unwanted(token):
        return False

    if count < min_count:
        if contains_emoji(token):
            return True
        else:
            return False

    elif count < max_count:
        if contains_emoji(token):
            return True
        else:
            if not_ascii(token):
                return False
            else:
                return True
    else:
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

kept = []
removed = []

print("remove_not_ok_pairs ...")
with Pool( processes = num_procs() ) as pool:
    for keep, remove in pool.imap_unordered(remove_not_ok_pairs, chunks):
        kept += keep
        removed += remove

print("sort kept pairs and removed pairs ...")
kept.sort( key = lambda x: -x[1] )
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


with open("tokens_removed.txt", "wt") as f:
    for tid, count in removed:
        f.write(pretty(tid, count) + "\n")


with open("tokens_kept.txt", "wt") as f:
    for tid, count in kept:
        f.write(pretty(tid, count) + "\n")


remains = set(wanted_tids)
for tid, _ in removed:
    tid = int(tid)
    if tid in remains:
        remains.remove(tid)

print(f"kept / total {len(kept)} / {tokenizer.vocab_size}")
print(f"remains / total {len(remains)} / {tokenizer.vocab_size}")


'''

python3 tokens_count.py 1000
wanted 105377 / 151643
stats_mode
1000 0
mkdir -p data
mkdir -p data/Qwen
mkdir -p data/Qwen/Qwen2.5-14B-Instruct
get_final_count ...
remove_not_ok_pairs ...
sort kept pairs and removed pairs ...

=== Một số removed tokens ===

70726      " bloque"                           999
96575      "emploi"                            999
77944      "Menus"                             999
71223      "plaintext"                         999
51648      ".deepcopy"                         999
73650      "(lo"                               999
40562      "='',"                              999
90094      "(rules"                            999
135972     " Glück"                            999
78955      "######\n"                          999
51423      "(pd"                               999
59100      "_Line"                             999
86632      " ------------------------------------------------------------"                                                                             999
13756      " tableView"                        999
4838       "semb"                              999
60288      "___\n\n"                           999
80915      ".logs"                             998
14039      "_NE"                               998
83686      "CONDITION"                         998
17560      " ;\r\n"                            998
49953      "_End"                              998
77784      " concatenate"                      998
68752      ")=\""                              998
28758      ".deltaTime"                        998
80612      "efeller"                           998
55735      "_segments"                         998
91578      "_resolver"                         998
79535      "(Group"                            998
83281      " wegen"                            998
74813      "\tpop"                             998
88387      "-pill"                             998
63645      " edx"                              998
62853      "TOKEN"                             997
76693      " vezes"                            997
30329      " prze"                             997
44765      ".ObjectId"                         997
57572      "_nm"                               997
47076      " progressBar"                      997
79790      "_First"                            997
70777      " Fragen"                           997
73999      " textbox"                          997
50732      "<Button"                           997
71381      "AsStream"                          997
49981      "erala"                             997
41472      "BUFFER"                            997
21672      " \\\r\n"                           997
87288      "_slices"                           997
21560      "(jPanel"                           997
93307      "ENTIC"                             996
85626      " requestBody"                      996
87220      " Tecn"                             996
36513      "_nr"                               996
98364      "(internal"                         996
35910      " BeautifulSoup"                    996
60623      " indexOf"                          996
46770      "DateString"                        996
67915      "=\\\"#"                            996
47135      "(reverse"                          996
60668      "_seen"                             996
65961      "/general"                          996
95538      " Chore"                            996
21668      " */\n\n\n"                         996
55033      "ADED"                              996
51288      " rowspan"                          996
76163      "ordova"                            996
69876      " gemacht"                          996
97158      "RenderingContext"                  996
80066      ".CurrentCulture"                   996
52894      " getters"                          996
61575      " numOf"                            996
78165      "entionPolicy"                      996
76077      ".hr"                               995
63448      " Hibernate"                        995
56246      "Signup"                            995
64206      " Laravel"                          995
62188      ".album"                            995
88737      " verifier"                         995
77397      " submenu"                          995
49527      "_MEDIA"                            995
75226      "HandlerContext"                    995
74354      " gemeins"                          995
43357      "_mc"                               995
56709      "EFI"                               995
50495      "_RULE"                             995
39219      "iveau"                             995
61373      "enqueue"                           995
51867      "_gallery"                          995
74239      ".band"                             995
57037      "_handlers"                         995
28148      "FXML"                              995
31068      "interop"                           994
28781      ".BorderStyle"                      994
63854      " także"                            994
88994      "ikut"                              994
35773      "_CALLBACK"                         994
34174      " TOD"                              994
72402      "xFFFFFF"                           994
97856      "(Unknown"                          994
64038      "_checksum"                         994
89567      " EventBus"                         994


72649      ".ModelSerializer"                  457
73718      "oldem"                             457
70731      "��"                                457
125030     "ıyor"                              457
92116      " PropertyChangedEventArgs"                                                                                                                 457
48080      "_para"                             457
97720      "_REPLACE"                          457
79829      " restTemplate"                     457
75470      "_WIFI"                             457
98762      "\tInit"                            456
76772      ".clientWidth"                      456
43018      "+\"</"                             456
48078      "ują"                               456
65731      "getWidth"                          456
91518      " ApplicationContext"               456
74942      "_IDS"                              456
44753      " scrollView"                       456
95435      ".checkbox"                         456
93775      " tileSize"                         456
99043      "_ATTRIB"                           456
86106      "HttpStatus"                        456
95727      ":''"                               456
26686      "htable"                            456
79769      "Arduino"                           456
75202      "QtCore"                            456
136443     " włas"                             456
83674      " createContext"                    456
94900      "_loan"                             456
49618      "\r\n\t\r\n"                        456
93190      "getRoot"                           456
79366      "_plate"                            456
50816      ":white"                            456
80320      " Gst"                              456
45432      "    \r\n    \r\n"                  456
67666      "_iface"                            456
93256      "ndx"                               455
90525      "setBackground"                     455
42925      " omp"                              455
63477      "\n    \n    \n"                    455
20355      "\\Controller"                      455
83697      "}}>"                               455
64265      "=\"\"\""                           455
71128      "\tloc"                             455
84237      " itm"                              455
79163      "basePath"                          455
76706      "=Math"                             455
90279      "_REDIRECT"                         455
99552      "�"                                 455
8150       "iminal"                            455
71630      " ()=>"                             455
80327      " menggunakan"                      455
63308      "getImage"                          455
57785      " removeFrom"                       455
81866      "(userid"                           455
93873      " ');\n\n"                          455
96803      " -------------"                    455
48402      "\tsetTimeout"                      455
90014      ".fits"                             455
87151      "kaar"                              455
83628      "\trandom"                          455
44401      "��"                                454
74833      "infile"                            454
83846      "/ros"                              454
18384      "@\","                              454
142037     " gehören"                          454
46013      " calloc"                           454
96428      " Mitarbeiter"                      454
93555      " Vaults"                           454
8803       " �"                                454
71739      "aporan"                            454
145109     "⦁"                                 454
62318      "(INFO"                             454
96869      ".ADMIN"                            454
55293      "PointerType"                       454
68714      ",obj"                              454
87161      "_Success"                          453
90873      "\tevents"                          453
144312     "◄"                                 453
43671      "ltk"                               453
89575      "WebHost"                           453
14048      "LOAT"                              453
48163      "_SR"                               453
83407      "_gps"                              453
75781      ":]:\n"                             453
57911      "(INPUT"                            453
89187      "_SUBJECT"                          453
82194      "GBK"                               453
76193      "ButtonType"                        453
80533      "LineStyle"                         453
75591      "nilai"                             453
75065      "GetEnumerator"                     453
88617      " ePub"                             452
95789      "_male"                             452
144979     "✈"                                 452
72469      ".clientX"                          452
45212      " StartCoroutine"                   452
93577      "_True"                             452
92932      "_MATERIAL"                         452
94883      "]){"                               452
140729     " täglich"                          452


70237      "+lsi"                                3
91599      "\"urls"                              3
87257      "_Printf"                             3
24338      "\r\r\r\n"                            3
76035      " datingside"                         3
65878      " thaimassage"                        3
143547     " mük"                                3
71195      " sexle"                              3
89637      "HomeAs"                              3
41864      "(EIF"                                3
89832      " analsex"                            3
96835      " sextreffen"                         3
95585      "]=]"                                 3
51796      "__(/*!"                              3
141399     " yönetici"                           3
143837     " vüc"                                3
82384      "_icall"                              3
92149      "();\r\r\n"                           3
97697      "APolynomial"                         3
81712      " beurette"                           3
143555     " müdah"                              3
14278      " /*<<<"                              2
61902      " sexdate"                            2
39957      " bakeca"                             2
139530     " günl"                               2
97971      "-cmpr"                               2
72128      "_InternalArray"                      2
86289      "\tUObject"                           2
127545     " düzenle"                            2
143839     " yayg"                               2
78508      "[MAXN"                               2
70270      " pornofil"                           2
24094      ":UIControl"                          2
90196      "CppI"                                2
142092     " pobli"                              2
83804      "gMaps"                               2
37735      "_StaticFields"                       2
140873     " jednocze"                           2
56669      "LIBINT"                              2
91239      "<UFunction"                          2
138173     "Cumhur"                              2
142582     " bölgesinde"                         2
98068      "SmartyHeaderCode"                    2
128171     "lararas"                             2
143855     " vazge"                              2
93552      "BracketAccess"                       2
84369      "/tinyos"                             2
76889      " \\<^"                               2
143452     " tecrübe"                            2
133118     " mücade"                             2
85791      " sidl"                               2
139210     " ülkem"                              2
84047      "Ubergraph"                           2
140324     " düzenlenen"                         2
139914     " Müdürü"                             1
78593      "_:*"                                 1
71918      " StreamLazy"                         1
24962      "methodVisitor"                       1
98372      " *}\n\n"                             1
23086      " [-]:"                               1
58739      "ConstraintMaker"                     1
81368      " uLocal"                             1
65509      "_hresult"                            1
126390     "maktad"                              1
89234      " pornofilm"                          1
128133     "prowadzi"                            1
52209      "arsimp"                              1
62685      " RTWF"                               1
56319      "_Statics"                            1
143838     " vücud"                              1
39170      "wcsstore"                            1
64792      " vivastreet"                         1
86278      " sexkontakte"                        1
132815     " mümk"                               1
88887      "lparr"                               1
88920      "rparr"                               1
44046      "%timeout"                            1
84962      "GameObjectWithTag"                   1
71507      "DECREF"                              1
56622      " ;;="                                1
90297      " [=["                                1
74161      "lbrakk"                              1
74084      "rbrakk"                              1
23543      "<lemma"                              1
78042      "\">';\r\n"                           1
52646      "yyval"                               1
50245      ")paren"                              1
133697     " Üniversites"                        1
139890     " sürecin"                            1
70290      " PodsDummy"                          1
140071     " gündem"                             1
138155     " seçen"                              1
136454     " sürek"                              1
143783     " uçu"                                1
49511      "VMLINUX"                             1
31283      " neuken"                             1
70266      "drFc"                                1
96481      " JSName"                             1
44694      ">tagger"                             1
91154      " XPAR"                               1


kept / total 79605 / 151643
remains / total 82388 / 151643
'''
