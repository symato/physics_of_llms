from ViTokenizer import tokenize
import lzma, json

for line in lzma.open("../data/test_pyvi.jsonl.xz"):
	data = json.loads(line)

	x = data["text"]
	assert tokenize(x, use_special_sep = False) == data["pyvi"]

	x = tokenize(x, use_special_sep = True)
	assert x.replace("▁", "_") == data["pyvi"]