FOCUS – Fast Overlapping Token Combinations Using Sparsemax
https://arxiv.org/pdf/2305.14481

FOCUS represents newly added tokens as combinations of tokens
in the overlap of the source and target vocabularies.

The overlapping tokens are selected
based on semantic similarity in an auxiliary static token embedding space.

https://github.com/konstantinjdobler/focus/blob/main/src/deepfocus/focus.py#L172
```py
'''
FOCUS method for transferring pretrained token embeddings to a different language from Dobler and de Melo (2023).

Args:
    target_tokenizer (PreTrainedTokenizer): The new tokenizer in the target language.

    source_tokenizer (PreTrainedTokenizer): The tokenizer for the pretrained source embeddings.

    source_embeddings (Tensor): The pretrained source embeddings tensor.

    auxiliary_embedding_mode ("fasttext-tokenlevel" or "fasttext-wordlevel"): The type of auxiliary embeddings to use. Defaults to "fasttext-tokenlevel".

    target_training_data_path (str | None, optional): Path to a file containing lines of text in the target language for training a fasttext model. Only necessary if using `fasttext-tokenlevel`. Defaults to None.

    fasttext_model_path (str | None, optional): Path to a pretrained fasttext model for the target tokenizer. Defaults to None.

    language_identifier (str | None, optional): Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.

    fasttext_model_epochs (int, optional): Number of epochs if training a custom fasttext model. Defaults to 3.
'''
    # Clean new tokens, mark "bad" tokens for random init
    random_init_new_tokens: list[NewToken] = []
    for token, new_token_info in tqdm(
        sorted_new_tokens,
        desc="Populating auxiliary embeddings for non-overlapping token...",
        leave=False,
    ):
        if is_very_rare_token(new_token_info.target.native_form, fasttext_model):
            random_init_new_tokens.append(new_token_info)
            del new_tokens[token]
        else:
            new_token_info.auxiliary_embedding = fasttext_model[token]

    ####################################################
    # 4. Copy source embeddings for overlapping tokens #
    ####################################################
    target_embeddings = torch.zeros((len(target_tokenizer), source_embeddings.shape[1]), device=device)
    for _, overlapping_token in sorted_overlapping_tokens:
        target_embeddings[overlapping_token.target.id] = overlapping_token.source_embedding

    ###########################################################
    # 5. Initialize "bad" new tokens from normal distribution #
    ###########################################################
    emb_mean = source_embeddings.mean(dim=0)
    emb_std = source_embeddings.std(dim=0)
    gen = torch.Generator(device=device).manual_seed(seed)
    for ood_new_token in random_init_new_tokens:
        target_embeddings[ood_new_token.target.id] = torch.normal(emb_mean, emb_std, generator=gen)

    #######################################################
    # 6. Finally, initialize additional tokens with FOCUS #
    #######################################################
    overlapping_tokens_for_focus = {k: v for k, v in sorted_overlapping_tokens if v.use_for_focus}
    target_embeddings = focus_additional_token_initialization(
        overlapping_tokens_for_focus, new_tokens, target_embeddings, device=device
    )
    logger.success(f"🎯 Initialized {len(new_tokens)} new tokens with FOCUS 🎯")
    return target_embeddings.detach()


def focus_additional_token_initialization(
    overlapping_tokens: dict[str, OverlappingToken],
    new_tokens: dict[str, NewToken],
    target_embeddings: Tensor,
    device: torch.device | str | None = None,
):
    # Convert to lists to ensure same order (`.values()` might not guarantee same order every time)
    new_tokens_lst = list(new_tokens.values())
    overlapping_tokens_lst = list(overlapping_tokens.values())

    # Convert to numpy arrays for fastdist
    new_auxiliary_embedding_matrix = np.asarray([t.auxiliary_embedding.tolist() for t in new_tokens_lst], dtype="float32")
    overlapping_auxiliary_embedding_matrix = np.asarray(
        [t.auxiliary_embedding.tolist() for t in overlapping_tokens_lst],
        dtype="float32",
    )

    logger.debug("Computing distance matrix...")
    similarity_matrix = fastdist.cosine_matrix_to_matrix(
        new_auxiliary_embedding_matrix,
        overlapping_auxiliary_embedding_matrix,
    )

    # Not needed anymore, save memory
    del new_auxiliary_embedding_matrix
    del overlapping_auxiliary_embedding_matrix

    logger.debug("Computing new embeddings...")

    # Do `torch.stack` once outside of loop to save time
    overlapping_src_embs = [t.source_embedding for t in overlapping_tokens_lst]
    overlapping_src_embs = torch.stack(overlapping_src_embs)

    for new_token_idx in tqdm(
        range(len(new_tokens_lst)),
        desc="FOCUS initialization...",
        total=len(new_tokens_lst),
    ):
        overlapping_emb_weights: Tensor = entmax.sparsemax(torch.from_numpy(similarity_matrix[new_token_idx]).to(device))

        # performance optimization
        mask = overlapping_emb_weights > 0.0
        masked_overlapping_emb_weights = overlapping_emb_weights[mask]
        masked_overlapping_src_embs = overlapping_src_embs[mask]

        weighted_src_embs = torch.mul(masked_overlapping_src_embs, masked_overlapping_emb_weights.unsqueeze(1))
        # It's a convex combination because the weights sum up to 1
        convex_combination = torch.sum(weighted_src_embs, dim=0)

        new_token_target_vocab_idx = new_tokens_lst[new_token_idx].target.id
        target_embeddings[new_token_target_vocab_idx] = convex_combination
    return target_embeddings
```

The key idea is to use overlapping tokens between both tokenizers as anchor points and represent
`new target language tokens` as a `weighted mean of over-lapping tokens' embeddings`.

Finding an initialization in the same semantic space as the pretrained embeddings is not as easy
for the set of `non-overlapping (“additional”) tokens`. To initialize embeddings for the 
additional tokens, we first train `auxiliary embeddings`for `all target tokens`.

We apply fastText on target language data pre-tokenized with the target tokenizer. Next, we compute the pairwise cosine similarities between the auxiliary embeddings of tokens in “additional” and “overlap”. Then `convert the similarity scores to weights by applying sparsemax`.
