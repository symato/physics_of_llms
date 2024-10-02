## Tỉa và mapping vocab sao cho model chỉ sinh ra tiếng Anh và Việt
- Các open source LLMs hiện tại khi dịch Anh, Việt thi thoảng model output tiếng Tàu
- Bộ từ vựng Anh Việt có lẽ chỉ chiếm 1/2 trong tổng số 100k - 200k vocab size.
  Khi tỉa gọn lại thì sẽ giúp:
  - Không thể ouptut tokens nào khác ngoài En hoặc Vi
  - Giảm vram khi infer và finetune embeddings
  - Tăng tốc độ infer và finetune

![](img/envi-405b-00.jpg)
*llama 3.1 405b vẫn hallu ra tiếng Trung trong tác vụ dịch Anh Việt*

**Cách làm**
- Tạo En, Vi dataset chứa các tài liệu tiếng Anh Việt và giả sử đó là toàn bộ nội dung của 2 ngôn ngữ

- Dùng En, Vi dataset trên chạy qua tokenizer để lọc lấy những tokens chỉ thuộc về En và Vi
  => Mảng `used_token_ids`

- Dồn embedding bằng cách map `used_token_ids[i] => i` (original token id => new token id)

**Tham khảo**
- https://github.com/huggingface/transformers/pull/31292
  - Llama3 8B -> MIND-BLOWING 3.62 memory usage reduction factor (due to large vocabulary)

- https://github.com/huggingface/transformers/issues/30860

- https://huggingface.co/5CD-AI/visocial-T5-base
  trimmed vocabulary size to 50,589 and continually pretrained google/mt5-base on a merged 20GB dataset

**Triển khai**
- [x] Dataset
- [x] Lọc theo thống kê mới giảm được gần một nửa 86k / 151k (qwen vocab)
  - bị mất một số emoji
- [x] Cần kết hợp với lọc theo bảng mã unicode
  - giữ lại emoji
  - loại bỏ cjk, thailand, chữ tượng hình ...
- [x] Target bộ từ vựng ~96k (63%)

- [x] Lọc sâu hơn nữa, target bộ từ vựng ~80k (50%)
  - [x] Giữ lại tokens chứa ký tự tiếng Việt
  - [x] Giữ lại ascii tokens

- [ ] Tạo final vocab
  - kept tokens @ `tokens_kept.txt`
  - [ ] special tokens (token điều khiển ...)

650GB text Việt, Anh, Code (100b tokens)
https://huggingface.co/datasets/Symonsters/NAM-005_436G_Vi-En-Code
```
 75.4G    100_book
 34.8G    100_code
 66.7G    100_slimpajama
- - - - - - - - - - - -
176.8G    English (40%)

118.9G    100_c4-vi
 11.3G    100_news_2020-2021
 15.1G    100_news_2022-2023
  2.6G    100_others
  2.0G    500_wiki
 20.3G    600_laws
 78.3G    800_epub
 11.3G    999_gov
- - - - - - - - - - 
259.8G    Vietnamese (60%)
    
Total 436.6GB, 100b tokens (Bloom tokenizer)

Theo thể loại:
 43%  187.6G   slimpajama, c4_vi, wiki
 35%  153.7G   book, epub
  8%   34.7G   code
  6%   26.4G   news
  5%   20.3G   laws
  3%   11.4G   gov
```

**Đối tượng thực hành**
- qwen2.5 có 0.5b, 1.5b, 3b, 7b, 14b, `32b`, `72b` models
- llama3.x có 1b, 3b, 8b, `70b`, 405b models
- gemma2 có 2b, 9b, `27b` models
- Các model được đánh giấu có chất lượng tốt và có thể quant để chạy trên 24G hoặc 40G vram

- [ ] Kiểm tra xem vocab của họ nhà qwen có giống nhau 100% không?
  - `qwen2.5` https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
  - `qwen2.0` https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat
  - `qwen1.5` https://huggingface.co/5CD-AI/Viet-Sailor-4B-Instruct

- [ ] Sửa code llama.cpp python hoặc exllama để có thể chạy đc model đã sửa vocab

- - -


## Sau khi tỉa gọn rồi từng bước một mở rộng bộ vocab

- tạo bộ từ điển từ ghép tiếng Việt thông dụng, chỉ cần khoảng 2k - 8k từ
- dùng một bộ lọc trước lúc tknz để lọc và map từ ghép này vào token id mới
- dùng một cách thông minh để khởi tạo embedding values của tokens mới
  - Zero-Shot Tokenizer Transfer https://github.com/bminixhofer/zett
- dùng lora finetune để refine new embeddings
  - freeze all layers, finetune embeddings trước
  - sau đó finetune all models
  - build datasets và giáo án huấn luyện phù hợp
  - ...
- ...

- - -

## Physics of LMs: làm lại thí nghiệm về  Knowledge Storage, Extraction and Manipulation

## Physics of LMs: cách tạo dữ liệu để chuyển giao knowledge từ En => Vi

