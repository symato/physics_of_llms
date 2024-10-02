```sh
head -n 3 tokens_kept__alphabet_long.jsonl
#  token                                         token_id  count
[  " opportunities"                               , 10488, 89723]
[  " technologies"                                , 14310, 47144]
[  " Pennsylvania"                                , 19771, 42463]
```

```sh
python3 tokens_count.py 800 20000
cd qwen__800__20000

## Sau khi cắt tỉa bằng máy
wc -l tokens_kept__*

    570 tokens_kept__alphabet_long.jsonl
  27227 tokens_kept__alphabet_short.jsonl
    343 tokens_kept__ascii_long.jsonl
  18297 tokens_kept__ascii_short.jsonl
  35092 tokens_kept__english.jsonl
   2536 tokens_kept__others.jsonl
  84065 total


## Sau khi cắt tỉa bằng tay
wc -l tokens_kept__*

    214 tokens_kept__alphabet_long.jsonl
  27227 tokens_kept__alphabet_short.jsonl
      0 tokens_kept__ascii_long.jsonl
  11047 tokens_kept__ascii_short.jsonl
  35092 tokens_kept__english.jsonl
   2536 tokens_kept__others.jsonl
  76116 total
```

Nhờ việc loại bỏ những tokens không thuộc hệ ngôn ngữ tiếng Anh + Việt (latin based), và loại bỏ những tokens có count thấp,
giảm được số lượng vocab của qwen từ 151k xuống ~80k (gần 1/2).

Việc phân loại các tokens vào từng mục như alphabet vs ascii vs other, long vs short giúp người xem cảm nhận tổng thể bộ vocab tốt hơn. Và có thể cắt tỉa thêm nếu muốn.