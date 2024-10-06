from ViTokenizer import tokenize
import lzma, json

for line in lzma.open("../data/test_pyvi.jsonl.xz"):
	data = json.loads(line)
	x = tokenize(data["text"])
	assert x == data["pyvi"]
