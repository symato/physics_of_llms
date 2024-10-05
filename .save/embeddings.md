- - -

From https://arxiv.org/pdf/2405.07883

Transfers LMs to a new tokenizer by initializing embedding parameters via a heuristic, then continuing to train the embeddings.

- init new embeddings value = mean old decoded embeddings 
  (the mean of the sequence of embeddings the new token is decomposed into by the previous tokenizer)

- RAMEN, WECHSEL, OFA require auxiliary embeddings ...

- FOCUS initializes embeddings of tokens in `Vb \ Va` as a weighted combination of the overlapping
tokens `Va ∩ Vb`, and **copies the embeddings of the overlapping tokens** ...

FOCUS obtains better performance without any training (i.e., zero-shot) than other heuristics.

Marchisio et al. (2023) show that forward- and backward-propagating through a `subset of the model layers` is sufficient for learning embeddings for a new tokenizer. Chen et al. (2023) find that `regularly resetting the embedding parameters during pretraining` boosts the speed at which they are relearnt upon transfer.

- - -

FOCUS key idea is to use overlapping tokens between both tokenizers as anchor points and represent
`new target language tokens` as a `weighted mean of over-lapping tokens' embeddings`.

Finding an initialization in the same semantic space as the pretrained embeddings is not as easy
for the set of `non-overlapping (“additional”) tokens`. To initialize embeddings for the 
additional tokens, we first train `auxiliary embeddings`for `all target tokens`.

We apply fastText on target language data pre-tokenized with the target tokenizer. Next, we compute the pairwise cosine similarities between the auxiliary embeddings of tokens in “additional” and “overlap”.
Then `convert the similarity scores to weights by applying sparsemax`.

> key idea là tính độ tương đồng giữa new tokens với existing tokens và dùng độ tương đồng đó và existing tokens' embeddings để khởi tạo.

- - -