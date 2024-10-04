import re
import torch
import transformers
import config

model_path = config.OFFLINE_MODEL_PATH
model = transformers.AutoModelForCausalLM.from_pretrained(
   model_path,
   torch_dtype = torch.bfloat16, # dtype gốc của qwen
   device_map = "cpu",
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
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

for w in sim_words :
	tids = tokenizer.encode(w)
	print(f"{w}{spaces[:maxx - len(w)]}{len(tids)} tokens {tids}", end = "\t\t")
	if len(tids) == 1:
		v = embeddings[tids[0]] # vector 1536 chiều
		print(v, v.shape)
	else:
		print()

''' Dùng umap để làm chiều không gian nhằm visualize cho dễ
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
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

'''