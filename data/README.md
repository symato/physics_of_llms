## Chuẩn bị data song ngữ Anh - Việt wikihow
```sh
# Lọc 93mb text song ngữ Anh - Việt, trong đó text tiếng Việt có độ dài lớn hơn 10k ký tự
python3 python3 prepare_wikihow_data.py 10000 > wikihow_more_filtered.jsonl

# Lọc 76mb text tiếng Việt từ wikipedia và wikisource
python3 prepare_wikimedia_data.py 3000 | shuf > wikimedia_vi.jsonl

# xem thử vài samples
head -n 10 wikihow_more_filtered.jsonl | jq
head -n 10 wikimedia_vi.jsonl | jq
```
