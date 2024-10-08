''' Luật merge của gemma (sentencepiece)

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

      "▁nh ững",
      "▁ những",  

      "▁như ng",
      "▁nh ưng",

      "▁Nh ững",
      "▁ Những",

      "▁Như ng",
      "▁Nh ưng",
      "▁ Nhưng",

      "▁như": 14105,
      "▁những": 15319,
      "▁nhưng": 34939,            
      "những": 71658,
      "Những": 98282,
      "▁Những": 116259,
      "▁Nhưng": 126294,
      "Nhưng": 168594,
      "▁Như": 211194,

'''

import os, sys, glob, json, math, lzma
# from utils_lang import *
from transformers import AutoTokenizer
from pprint import pprint

a = json.load(lzma.open("gemma_tokenizer.json.xz"))

for k, v in a.items():
    if isinstance(v, list):
        print(k) # added_tokens only
        pprint(v[:3])
        print(k) # added_tokens only
print()

print("pre_tokenizer", a["pre_tokenizer"])
print()

print("post_processor", a["post_processor"])
print()

print(a.keys())
print()

print("decoder", a["decoder"])
print()

print("model", a["model"].keys())
print()

model = a["model"]
print("model vocab", list(model["vocab"].items())[10000:10010])
print()

print("model merges", model["merges"][10000:10010])
print()

'''
"▁St" "ates" => "▁States"

"▁St": 997,
"ates": 1204,
"▁States": 3858,
'''

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
def tknz(text):
    tids = tokenizer.encode(text)[1:]
    tokens = [ tokenizer.decode(x) for x in tids ]
    return tokens

print(tknz(" States "))
print(tknz(" St ates "))
print(tknz(" St |ates "))
