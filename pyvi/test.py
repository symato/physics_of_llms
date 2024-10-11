from ViTokenizer import tokenize, tknz
import lzma, json

for idx, line in enumerate( lzma.open("../data/test_pyvi.jsonl.xz") ):
	data = json.loads(line)

	x = tokenize(data["text"], use_special_sep = False)
	assert x == data["pyvi"]

	x = tokenize(data["text"], use_special_sep = True)
	assert x.replace("▁", "_") == data["pyvi"]

	x = tknz(data["text"])
	assert x.replace("▁", " ") == data["text"]

	if idx == 3:
		print(tokenize(data["text"], use_special_sep = False))
		print(x)
