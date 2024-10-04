from transformers import AutoTokenizer
import subprocess, os, sys

from utils_lang import *
from utils import num_procs

from multiprocessing import Pool
import glob, json


try: do_check_lang = sys.argv[1] == "bylang"
except: do_check_lang = False
print("do_check_lang", do_check_lang)

from config import ONLINE_MODEL_PATH as model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)

if do_check_lang:
    subprocess.run("rm -rf data/tokens_by_lang", shell = True)
    subprocess.run("mkdir -p data/tokens_by_lang", shell = True)

    def check_check(reg, token):
        m = regex.findall(reg, token)
        for x in m:
            for c in x:
                if ord(c) > 255: # not ascii
                    return True
        return False


    def write_to_lang_file(lang, tid, token):
        filename = f"data/tokens_by_lang/{lang}.jsonl"
        with open(filename, "at") as f:
            f.write(json.dumps({"tid": tid, "token": token}, ensure_ascii = False) + "\n")


    def check_for_cjk_vi(lang, tid, token):
        if contains_cjk(token):
            write_to_lang_file("CJK", tid, token)

        elif canbe_vietnamese(token):
            write_to_lang_file("CanBeVietnamese", tid, token)
        else:
            write_to_lang_file(lang, tid, token)


    unwanted_lang_re_pairs = {"Latin": regex.compile(f'[\pLatin]+')}

    for x in unwanted_langs:
        unwanted_lang_re_pairs[x[3 : -1]] = regex.compile(f'[{x}]+')


    def check_vocab(lang_reg):
        lang, reg = lang_reg

        for tid in range(0, tokenizer.vocab_size):
            token = tokenizer.decode(tid)

            if check_check(reg, token):
                if not contains_cjk(token) and not canbe_vietnamese(token):
                    write_to_lang_file(lang, tid, token)

    with Pool( processes = num_procs() ) as pool:
        for _ in pool.imap_unordered(check_vocab, unwanted_lang_re_pairs.items()):
            pass

    processed_tids = []
    for filename in glob.glob("data/tokens_by_lang/*"):
        processed_tids += [ json.loads(line)["tid"] for line in open(filename, "rt") ]
    processed_tids = set( processed_tids )

    for tid in range(0, tokenizer.vocab_size):
        if tid not in processed_tids: # Others
            token = tokenizer.decode(tid)
            check_for_cjk_vi("Others", tid, token)            

else:

    wanted = open("data/tokens_wanted.txt", "wt")
    unwanted = open("data/tokens_unwanted.txt", "wt")
    wanted_tids = []

    for tid in range(0, tokenizer.vocab_size):
        token = tokenizer.decode(tid)

        if contains_unwanted(token):
            unwanted.write(token + "\n")
        else:
            wanted_tids.append(tid)
            wanted.write(token + "\n")

    print(f"wanted {len(wanted_tids)} / {tokenizer.vocab_size}")
