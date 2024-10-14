## Chuẩn bị data để tiếp tục training LLM
```sh
# Lọc ~90mb text song ngữ Anh - Việt, trong đó text tiếng Việt có độ dài lớn hơn 10k ký tự
python3 prepare_wikihow_data.py 3000 24000 | shuf > wikihow_vien_filtered.jsonl

# Lọc ~110mb text tiếng Việt từ wikipedia và wikisource
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikisource__20231201.vi__train-00.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-00a.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-00b.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-00c.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-01.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-02.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-03a.jsonl.xz

python3 prepare_wikimedia_data.py 3000 | shuf > wikimedia_vi_filtered.jsonl

# ~100mb text song ngữ mini-vncc-envi_30k.jsonl
python3 prepare_mini-vncc-envi.py > mini-vncc-envi.jsonl

# xem thử vài samples
shuf wikihow_vien_filtered.jsonl | head -n 10 | jq
shuf wikimedia_vi_filtered.jsonl | head -n 10 | jq
```

## Chuẩn bị instructions data
```sh
wget https://huggingface.co/datasets/ssmi153/Capybara-ShareGPT/resolve/main/CapybaraPure_Decontaminated.jsonl

wget https://huggingface.co/datasets/Gryphe/Sonnet3.5-SlimOrcaDedupCleaned-20k/resolve/main/sharegpt_claude_sonnet3.5_slimorca_subset.jsonl

./prepare_webglm-qa.py THUDM__webglm-qa__train__a__vi.jsonl.xz > webglm-qa_vien.jsonl
```

## Final finetune data
```sh
cat wikihow_vien_filtered.jsonl mini-vncc-envi.jsonl webglm-qa_vien*.jsonl \
	wikimedia_vi_filtered.jsonl \
	 \
	sharegpt_claude_sonnet3.5_slimorca_subset.jsonl CapybaraPure_Decontaminated.jsonl \
	| shuf > final_finetune1.jsonl

du -sh final_finetune1.jsonl # 793M

du -sh *.jsonl
# Dữ liệu instructs tiếng Anh
72M     CapybaraPure_Decontaminated.jsonl
65M     sharegpt_claude_sonnet3.5_slimorca_subset.jsonl

# Dữ liệu RAG / QA Anh => Việt, Việt => Việt
45M     webglm-qa_vien_a.jsonl
44M     webglm-qa_vien_b.jsonl
45M     webglm-qa_vien_c.jsonl
42M     webglm-qa_vien_d.jsonl

# Dữ liệu dịch song ngữ
92M     mini-vncc-envi.jsonl
92M     wikihow_vien_filtered.jsonl

# Dữ liệu thuần Việt
300M    wikimedia_vi_filtered.jsonl
```
