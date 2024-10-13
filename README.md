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
- [x] Tạo final vocab từ [qwen__1000__20000](./qwen__1000__20000/README.md)

**Đối tượng thực hành**
- qwen2.5 có 0.5b, 1.5b, 3b, 7b, 14b, `32b`, `72b` models
- llama3.x có 1b, 3b, 8b, `70b`, 405b models
- gemma2 có 2b, 9b, `27b` models
- Các model được đánh giấu có chất lượng tốt và có thể quant để chạy trên 24G hoặc 40G vram

- [ ] Kiểm tra xem vocab của họ nhà qwen có giống nhau 100% không?
  - `qwen2.5` https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
  - `qwen2.0` https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat
  - `qwen1.5` https://huggingface.co/5CD-AI/Viet-Sailor-4B-Instruct

- [x] Thêm special tokens vào `qwen__1000__20000/tokens_kept__*` để tạo new vocab

- [x] Thử cắt tỉa qwen2.5 1.5b và chạy inference

- [ ] Sửa code llama.cpp python hoặc exllama để có thể chạy đc model đã sửa vocab

```sh
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ../Qwen2.5-1.5B-Instruct

python3 model_edit.py -m ../Qwen2.5-1.5B-Instruct -t trimm_vocab

python3 model_chat.py ../Qwen2.5-1.5B-Instruct
# Bạn: Translate following sentence into Chinese: tôi tên là Lý Quốc Dân
# Bot: 我的名字是李国定
# ../Qwen2.5-1.5B-Instruct: timespent 1.31 seconds

python3 model_chat.py ../Qwen2.5-1.5B-Instruct__trimm_vocab
# Bạn: Translate following sentence into Chinese: tôi tên là Lý Quốc Dân
# Bot: My name is Li Guo Dan.
# ../Qwen2.5-1.5B-Instruct__trimm_vocab: timespent 1.31 seconds
```
Vì bộ vocab mới chỉ hỗ trợ tiếng Anh và Việt là chính nên qwen không thể nói tiếng Trung được nữa

- - -

## Physics of LMs: Sau khi tỉa gọn, từng bước một mở rộng vocab

- [x] tạo bộ từ điển từ ghép tiếng Việt thông dụng và chiếm nhiều tokens nhất, chỉ cần khoảng < 10k
  - Dùng https://github.com/trungtv/pyvi để tách từ
```sh
# `impact = freq * (qwen_tokens_count - 1)` (`freq` là tần suất sử dụng của từ đó trong corpus)
xzcat data/vi_words_impact.jsonl.xz | head -n 10
{"word": "có▁thể",     "impact": 328134, "qwen_tokens": ["có", " thể"], "freq": 328134, "qwen_tokens_count": 2}
{"word": "sử▁dụng",    "impact": 290336, "qwen_tokens": ["s", "ử", " dụng"], "freq": 145168, "qwen_tokens_count": 3}
{"word": "nghiên▁cứu", "impact": 251019, "qwen_tokens": ["n", "ghi", "ên", " cứu"], "freq": 83673, "qwen_tokens_count": 4}
{"word": "thời▁gian",  "impact": 210258, "qwen_tokens": ["th", "ời", " g", "ian"], "freq": 70086, "qwen_tokens_count": 4}
{"word": "Việt▁Nam",   "impact": 197542, "qwen_tokens": ["Vi", "ệt", " Nam"], "freq": 98771, "qwen_tokens_count": 3}
{"word": "Tại▁sao",    "impact": 163515, "qwen_tokens": ["T", "ại", " sa", "o"], "freq": 54505, "qwen_tokens_count": 4}
{"word": "hoạt▁động",  "impact": 154496, "qwen_tokens": ["ho", "ạt", " động"], "freq": 77248, "qwen_tokens_count": 3}
{"word": "một▁số",     "impact": 137332, "qwen_tokens": ["m", "ột", " số"], "freq": 68666, "qwen_tokens_count": 3}
{"word": "thực▁hiện",  "impact": 131574, "qwen_tokens": ["th", "ực", " hiện"], "freq": 65787, "qwen_tokens_count": 3}
{"word": "đầu▁tiên",   "impact": 123062, "qwen_tokens": ["đ", "ầu", " tiên"], "freq": 61531, "qwen_tokens_count": 3}

xzcat data/vi_words_impact.jsonl.xz | head -n 1000 | tail -n 10 
{"word": "riêng▁tư",   "impact": 8163, "qwen_tokens": ["ri", "ê", "ng", " tư"], "freq": 2721, "qwen_tokens_count": 4}
{"word": "Đặc▁biệt",   "impact": 8160, "qwen_tokens": ["Đ", "ặc", " biệt"], "freq": 4080, "qwen_tokens_count": 3}
{"word": "Tổng▁cục",   "impact": 8157, "qwen_tokens": ["T", "ổ", "ng", " cục"], "freq": 2719, "qwen_tokens_count": 4}
{"word": "tiêu▁hóa",   "impact": 8142, "qwen_tokens": ["ti", "êu", " hóa"], "freq": 4071, "qwen_tokens_count": 3}
{"word": "Hải▁Phòng",  "impact": 8128, "qwen_tokens": ["H", "ải", " Phòng"], "freq": 4064, "qwen_tokens_count": 3}
{"word": "bố▁trí",     "impact": 8122, "qwen_tokens": ["b", "ố", " trí"], "freq": 4061, "qwen_tokens_count": 3}
{"word": "ý▁kiến",     "impact": 8104, "qwen_tokens": ["ý", " kiến"], "freq": 8104, "qwen_tokens_count": 2}  
{"word": "định▁cư",    "impact": 8086, "qwen_tokens": ["đ", "ịnh", " cư"], "freq": 4043, "qwen_tokens_count": 3}
{"word": "hải▁quân",   "impact": 8078, "qwen_tokens": ["h", "ải", " quân"], "freq": 4039, "qwen_tokens_count": 3}
{"word": "tư▁tưởng",   "impact": 8076, "qwen_tokens": ["t", "ư", " tưởng"], "freq": 4038, "qwen_tokens_count": 3}

xzcat data/vi_words_impact.jsonl.xz | head -n 2000 | tail -n 10 
{"word": "cơ▁học",     "impact": 3624, "qwen_tokens": ["c", "ơ", " học"], "freq": 1812, "qwen_tokens_count": 3}
{"word": "luôn▁luôn",  "impact": 3620, "qwen_tokens": ["lu", "ôn", " luôn"], "freq": 1810, "qwen_tokens_count": 3}
{"word": "tâm▁trạng",  "impact": 3618, "qwen_tokens": ["t", "âm", " trạng"], "freq": 1809, "qwen_tokens_count": 3}
{"word": "Động▁vật",   "impact": 3616, "qwen_tokens": ["Đ", "ộng", " vật"], "freq": 1808, "qwen_tokens_count": 3}
{"word": "Xô▁viết",    "impact": 3616, "qwen_tokens": ["X", "ô", " viết"], "freq": 1808, "qwen_tokens_count": 3}
{"word": "Tỷ▁lệ",      "impact": 3614, "qwen_tokens": ["T", "ỷ", " lệ"], "freq": 1807, "qwen_tokens_count": 3}
{"word": "thực▁thể",   "impact": 3614, "qwen_tokens": ["th", "ực", " thể"], "freq": 1807, "qwen_tokens_count": 3}
{"word": "giả▁định",   "impact": 3614, "qwen_tokens": ["gi", "ả", " định"], "freq": 1807, "qwen_tokens_count": 3}
{"word": "truyền▁tải", "impact": 3612, "qwen_tokens": ["tr", "uyền", " tải"], "freq": 1806, "qwen_tokens_count": 3}
{"word": "mạch▁máu",   "impact": 3606, "qwen_tokens": ["m", "ạch", " máu"], "freq": 1803, "qwen_tokens_count": 3}
```
![](img/vi-words-impact-02.jpg)

=> **Chọn khoảng 1k - 2k từ ghép để mở rộng vocab là đủ tạo impact**

- [ ] lọc và map những từ ghép này vào token ids mới

- [ ] tìm các cách *hiệu quả* để khởi tạo embedding values của tokens mới
  - Với 1 từ được chọn, tìm ra 1-3 câu liên quan tới từ đó:
    - thay toàn bộ embedding values của từ được chọn băng 01 embedding value mới được init bằng nhiều cách:
      - embedding value của từ đơn tương ứng trong tiếng Anh
      - lấy trung bình cộng của các embedding values của các tokens của từ đó
      - lấy trung bình cộng của toàn bộ embedding values của các từ `gần` với nó (trong TV và các ngôn ngữ khác)
        `gần` ở đây có thể là về ý nghĩa, về embdding values hoặc bất kỳ độ đo hợp lý nào ...
  - làm thế nào để đo lường được *hiệu quả*?
    - tính sự khác biệt của output (logits diff / perpelexity ...) trong các phép thay thế,
    khác biệt thấp nhất => hiệu quả nhất?
  - Việc lựa chọn embedding values có thực sự quan trọng? Vì đằng nào cũng cần continue pretrain.

- [ ] Mát xa new embeddings (and old embeddings too)
  - freeze *most* layers, finetune embeddings và vài low layers trước
    - Lý do: từ vựng, ngữ pháp, các skills ngôn ngữ tập trung nhiều ở low layers
  - sau đó finetune toàn bộ model (lora + embedding or full finetune)
  - build datasets và giáo án huấn luyện phù hợp

- - -

Bài toán: cho một từ (ví dụ "thực hiện") là thế nào để tìm ra một embding value mà khi dùng nó để thay thế chuỗi tokens
"qwen_tokens": [" thực", " hiện"] trong các đoạn text mà nó xuất hiện thì không làm thay đổi `đầu ra` của model.

Đầu ra ở đây là một giá trị càng gần 0 càng tốt (0 = không thay đổi), có thể là logits diff hoặc perplexity, 

Bạn: tạo ra một câu hoàn chỉnh với từ "thực hiện"
Bot: "Tôi đã thực hiện kế hoạch của mình thành công."

Giờ ta sẽ mask từ "thực hiện" và được "Tôi đã ___ kế hoạch của mình thành công.", 
giờ ta để LLM tự điền vào chỗ trống 01 token thì liệu nó có tìm ra token có embedding value hợp lý nhất cho từ "thực hiện" không?

Hidden value (embedding) ở layer cuối, khi nhân với lm_head để tạo logits và chọn ra vị trí có logits cao nhất làm token_id, 
lm_head value ở ví trí đó với qwen 1.5 chính là embedding value vì qwen 1.5 dùng tied embeddings.

![](img/vocab-extend-00.jpg)

![](img/vocab-extend-02.jpg)

- - -

## Kỹ thuật dồn vocab khi finetune

Khi finetune trên 1 tập domain data nhỏ (vài GB) sẽ không dùng hết 100k - 200k vocab, => 
lọc tokens thực sự dùng trong data (1/4 - 1/2), dồn embeddings lại và chỉ train trên những embeddings đó.
Sau khi train xong lại re-map và merge vào vocab gốc.

Ưu: 
- Save vram while training! (embeddings ở định dạng f32 nên khá tốn)
- không làm ảnh hưởng bộ tknz gốc

Nhược:
- Các embeddings khác không được tune?

- - -

## Thử nghiệm In-context Pretraining xem khi trộn lẫn training sample trong ctxlen ảnh hưởng gì tới model?
- trộn ngẫu nhiên
- trộn có tính toán https://arxiv.org/abs/2310.10638
- không trộn (mỗi sample 1 ctxlen hoặc dùng packed dataset)

## Thử nghiệm Block Expansion
- https://arxiv.org/abs/2401.02415v2

## Thử nghiệm LAYER SWAPPING
- https://arxiv.org/abs/2410.01335

## Physics of LMs: làm thí nghiệm về Knowledge Storage, Extraction and Manipulation

## Physics of LMs: làm thí nghiệm [TinyStories](TinyStories.md) về học language
- Build dataset theo một hướng khác? TinyFantasy? TinyFunny?
- Mở rộng: xây bộ data để chuyển knowledge đã học từ Vi => En
- Mở rộng: Từ hiểu ngôn ngữ tới làm thơ và làm thơ thuận nghịch độc
  - https://nhathongnguyenthanhvan.wordpress.com/2018/03/20/nhung-bai-tho-thuan-nghich-doc
  - Nên bắt đầu với dataset thuận nghịch độc, các câu ngắn đọc xuôi hay ngược đều có ý nghĩa
- Ý tưởng: dùng kỹ thuật FIM (fill in middle) của code LLM để tạo thơ có vần
  Yêu cầu chữ cuối phải rơi vào 1 từ hoặc 1 vần nào đó ...

![](img/tho-thuan-nghich-doc.jpg)
