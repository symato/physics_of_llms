import re
import torch
import transformers
import config
import os, json, re, sys
from pprint import pprint

model_path = config.OFFLINE_MODEL_PATH
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

sim0 = [ json.loads(line) for line in open("data/vi_words_similarity.jsonl") ]
sim1 = [ json.loads(line) for line in open("data/vi_words_similarity_berua.jsonl") ]

# pprint(sim0[0])
# pprint(sim1[0]); input()


def get_similiar_words(n = None):
	words = {}

	def get_uniq_token_ids(word, en_word):

		if word not in words:
			words[word] = { }

		variants = [
			en_word,
			en_word.lower(),

			" " + en_word,
			" " + en_word.lower()
		]

		for x in variants:
			tids = tokenizer.encode(x)
			if len(tids) == 1: # chỉ lấy en_word có 1 token
				words[word][x] = tids[0]


	for x in sim0:
		word = x["word"]
		text = x["textbook"]
		splits = re.split(r'\d. "(.+?)[/"]', text)
		
		for i in range(1, len(splits), 2):
			en_word = splits[i].strip()
			get_uniq_token_ids(word, en_word)


	for x in sim1:
		word = x["term"]
		for e in x["example"]:
			en_word = e["term"]
			# get_uniq_token_ids(word, en_word)

	# Loại bỏ words 
	for word, en_words in list(words.items()):
		if len(en_words) == 0:
			words.pop(word)

	words = list(words.items())
	if n is None:
		return words
	else:
		return words[ : n]


if __name__ == "__main__":
	words = get_similiar_words()
	pprint(words)
	print(len(words))


'''
model = transformers.AutoModelForCausalLM.from_pretrained(
   model_path,
   torch_dtype = torch.bfloat16, # dtype gốc của qwen
   device_map = "cpu",
)
embeddings = model.model.embed_tokens.weight.detach()

sim_words = """
https://chatgpt.com/share/66ffece5-7504-8001-900c-26b5acd46a9d

Dưới đây là 10 từ đồng nghĩa của "thực hiện":

1. Tiến hành
2. Thi hành
3. Thực thi
4. Thực hành
5. Đảm nhận
6. Áp dụng
7. Hoàn thành
8. Triển khai
9. Thiết lập
10. Cử hành

Here is the translation of the synonyms for "thực hiện" into English:

1. Conduct
2. Enforce
3. Execute
4. Implement
5. Undertake
6. Apply
7. Complete
8. Deploy
9. Establish
10. Perform
""".split("\n")

sim_words = [ x.split('.')[1].lower() for x in sim_words if re.match(r'\d\. ', x) ]

maxx = 20
spaces = " " * ( maxx + 1 )

for w in sim_words:
	tids = tokenizer.encode(w)
	print(f"{w}{spaces[:maxx - len(w)]}{len(tids)} tokens {tids}", end = "\t\t")
	if len(tids) == 1:
		v = embeddings[tids[0]] # vector 1536 chiều
		print(v, v.shape)
	else:
		print()
'''

'''
 conduct    1 tokens [6786]     embedding([ 0.0040,  0.0184, -0.0278,  ...,  0.0092, -0.0757, -0.0282],
 enforce    1 tokens [28162]    embedding([ 0.0532, -0.0021,  0.0154,  ...,  0.0167, -0.0152,  0.0198],
 execute    1 tokens [9026]     embedding([ 0.0342, -0.0303,  0.0291,  ...,  0.0410,  0.0142,  0.0124],
 implement  1 tokens [4211]     embedding([ 0.0130,  0.0300,  0.0374,  ...,  0.0145,  0.0254,  0.0205],
 undertake  1 tokens [48543]    embedding([ 0.0505,  0.0339, -0.0055,  ...,  0.0071, -0.0142, -0.0059],
 apply      1 tokens [3796]     embedding([ 0.0171,  0.0300, -0.0232,  ...,  0.0175,  0.0034, -0.0125],
 complete   1 tokens [4583]     embedding([ 0.0427, -0.0322,  0.0356,  ..., -0.0090,  0.0216, -0.0063],
 deploy     1 tokens [10517]    embedding([-0.0024, -0.0065, -0.0204,  ...,  0.0474,  0.0225,  0.0039],
 establish  1 tokens [5695]     embedding([ 0.0498, -0.0374,  0.0038,  ...,  0.0073,  0.0166, -0.0083],

Cần visualize để "nhìn" được sự giống nhau giứa các embeddings

Bài toán: cho một từ (ví dụ "thực hiện") là thêm nào để tìm ra một embding value mà khi dùng nó để thay thế chuỗi tokens
"qwen_tokens": [" thực", " hiện"] trong các đoạn text mà nó xuất hiện thì không làm thay đổi `đầu ra` của model.

Đầu ra ở đây là một giá trị càng gần 0 càng tốt (0 = không thay đổi), có thể là logits diff hoặc perplexity, 

Bạn: tạo ra một câu hoàn chỉnh với từ "thực hiện"
Bot: "Tôi đã thực hiện kế hoạch của mình thành công."

Giờ ta sẽ mask từ "thực hiện" và được "Tôi đã ___ kế hoạch của mình thành công.", 
giờ ta để LLM tự điền vào chỗ trống 01 token thì liệu nó có tìm ra token có embedding value hợp lý nhất cho từ "thực hiện" không?

Hidden value (embedding) ở layer cuối, khi nhân với lm_head để tạo logits và chọn ra vị trí có logits cao nhất làm token_id, 
lm_head value ở ví trí đó với qwen 1.5 chính là embedding value vì qwen 1.5 dùng tied embeddings.


Dùng umap để làm chiều không gian nhằm visualize cho dễ
reduced_embeddings = umap.UMAP(
    n_neighbors=n_neighbors, n_components=dim, metric=metric
).fit_transform(embeddings)
'''
