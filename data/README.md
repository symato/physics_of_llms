## Chuẩn bị data để tiếp tục training LLM
```sh
# Lọc ~90mb text song ngữ Anh - Việt, trong đó text tiếng Việt có độ dài lớn hơn 10k ký tự
python3 prepare_wikihow_data.py 6000 24000 | shuf > wikihow_vien_filtered.jsonl

# Lọc ~110mb text tiếng Việt từ wikipedia và wikisource
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikisource__20231201.vi__train-00.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-00b.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-01.jsonl.xz
wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-02.jsonl.xz
# wget https://huggingface.co/datasets/Symato/KB_wikimedia/resolve/main/wikipedia__20231101.vi__train-03b.jsonl.xz

python3 prepare_wikimedia_data.py 3000 | shuf > wikimedia_vi_filtered.jsonl

# xem thử vài samples
shuf wikihow_vien_filtered.jsonl | head -n 10 | jq
shuf wikimedia_vi_filtered.jsonl | head -n 10 | jq
```
