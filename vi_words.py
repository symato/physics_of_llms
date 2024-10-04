import lzma, json

most_uncompressed_vi_words = [ json.loads(line) for line in lzma.open("data/vi_words_score.jsonl.xz") ]

print(most_uncompressed_vi_words)