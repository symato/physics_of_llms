import lzma, json
from pyvi import ViTokenizer # pip install pyvi

for idx, line in enumerate(lzma.open("test.jsonl.xz")):
	if idx >= 150: break

	text = json.loads(line)["text"]
	x = ViTokenizer.tokenize(text)

	print(json.dumps({
		"text": text,
		"pyvi": x,
	}, ensure_ascii = False))
