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

**Tham khảo tác dụng của giảm tải vocab và can thiệp vào inference**
- https://github.com/huggingface/transformers/pull/31292
  - Llama3 8B -> MIND-BLOWING 3.62 memory usage reduction factor (due to large vocabulary)

- https://github.com/huggingface/transformers/issues/30860

- https://huggingface.co/5CD-AI/visocial-T5-base
  trimmed vocabulary size to 50,589 and continually pretrained google/mt5-base on a merged 20GB dataset

- https://github.com/sam-paech/antislop-sampler given a list of words & phrases to avoid like 
  "a tapestry of", "a testament to", etc., and it will backtrack and try something else if it hits that phrase.

**Triển khai**
- [x] Dataset
- [x] Lọc theo thống kê mới giảm được gần một nửa (qwen vocab)
  - bị mất một số emoji
- [x] Cần kết hợp với lọc theo bảng mã unicode
  - giữ lại emoji
  - loại bỏ cjk, thailand, chữ tượng hình ...
- [x] Target bộ từ vựng ~96k (63%)

- [x] Lọc sâu hơn nữa, target bộ từ vựng ~80k (50%)
  - [x] Giữ lại tokens chứa ký tự tiếng Việt
  - [x] Giữ lại ascii tokens

- [x] Tạo final vocab từ [qwen__800__20000](./qwen__800__20000/README.md)
  - [x] Lọc tay kept tokens @ `tokens_kept__*.jsonl` (84065 giảm xuống 76116)

**Đối tượng thực hành**
- qwen2.5 có 0.5b, 1.5b, 3b, 7b, 14b, `32b`, `72b` models
- llama3.x có 1b, 3b, 8b, `70b`, 405b models
- gemma2 có 2b, 9b, `27b` models
- Các model được đánh giấu có chất lượng tốt và có thể quant để chạy trên 24G hoặc 40G vram

- [ ] Kiểm tra xem vocab của họ nhà qwen có giống nhau 100% không?
  - `qwen2.5` https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
  - `qwen2.0` https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat
  - `qwen1.5` https://huggingface.co/5CD-AI/Viet-Sailor-4B-Instruct

- [x] Thêm special tokens vào `qwen__800__20000/tokens_kept__*` để tạo new vocab

- [x] Thử cắt tỉa qwen2.5 1.5b và chạy inference

- [ ] Sửa code llama.cpp python hoặc exllama để có thể chạy đc model đã sửa vocab

```sh
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ../Qwen2.5-1.5B-Instruct

python3 qwen_edit.py ../Qwen2.5-1.5B-Instruct

python3 qwen_chat.py ../Qwen2.5-1.5B-Instruct
```
```
new_vocab 76138
Bạn: bạn tên gì?

Tôi là một trợ lý ảo, không có tên riêng.
qwenchat: timespent 0.65 seconds

Bạn: dịch đoạn văn sau sang tiếng Anh: tôi tên là Lý Quốc Dân

Here's the English translation of the Vietnamese text:

"My name is Ly Quoc Dan."

This directly translates the given Vietnamese sentence into English, maintaining its original meanriginal meaning and structure.
qwenchat: timespent 1.15 seconds

Bạn: dịch đoạn văn sau sang tiếng Trung: tôi tên là Lý Quốc Dân

Here's the English translation of the Vietnamese text:

Mr. Tran: My name is Le Quoc Dan.
qwenchat: timespent 0.98 seconds
```
Vì bộ vocab mới chỉ hỗ trợ tiếng Anh và Việt là chính nên qwen không thể nói tiếng Trung được nữa


- - -


## Sau khi tỉa gọn rồi từng bước một mở rộng bộ vocab

- [ ] tạo bộ từ điển từ ghép tiếng Việt thông dụng, chỉ cần khoảng 2k - 8k từ thông dụng nhất
  - Dùng https://github.com/trungtv/pyvi để tách từ ghép

- dùng một bộ lọc trước lúc tknz để lọc và map từ ghép này vào token id mới
- dùng một cách thông minh để khởi tạo embedding values của tokens mới
- Vỗ về new embeddings
  - freeze all layers, finetune embeddings trước
  - sau đó finetune models (lora + embedding or full finetune)
  - build datasets và giáo án huấn luyện phù hợp
- ...


- - -


## Physics of LMs: làm thí nghiệm về Knowledge Storage, Extraction and Manipulation

## Physics of LMs: làm thí nghiệm [TinyStories](TinyStories.md) về học language
- Build dataset theo một hướng khác? TinyFantasy? TinyFunny?
- Mở rộng: xây bộ data để chuyển knowledge đã học từ En => Vi

**Prompt**: viết truyện cười về xe ô tô và người lập trình viên

**Response**: Một lập trình viên mua chiếc xe ô tô mới và rất tự hào. Anh ta quyết định đưa nó đến khoe với đồng nghiệp tại công ty phần mềm.
Khi đến nơi làm việc, anh ta hào hứng nói với mọi người: "Này các bạn, tôi vừa mua một chiếc xe mới toanh! Nó có thể tự động lái, tự đỗ xe, và thậm chí còn có thể nói chuyện với tôi nữa!"
Một đồng nghiệp tò mò hỏi: "Ồ, vậy bạn đã thử nói chuyện với nó chưa?"
Lập trình viên trả lời: "Rồi, nhưng nó cứ báo lỗi 'Syntax Error' mỗi khi tôi nói tiếng người!"
