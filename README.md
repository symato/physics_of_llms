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
{"word": " có▁thể",      "impact": 12124202, "qwen_tokens": [" có", " thể"], "freq": 12124202, "qwen_tokens_count": 2}
{"word": " thời▁gian",   "impact": 8537648, "qwen_tokens": [" thời", " g", "ian"], "freq": 4268824, "qwen_tokens_count": 3}
{"word": " Việt▁Nam",    "impact": 7424085, "qwen_tokens": [" Việt", " Nam"], "freq": 7424085, "qwen_tokens_count": 2}
{"word": " thực▁hiện",   "impact": 5479125, "qwen_tokens": [" thực", " hiện"], "freq": 5479125, "qwen_tokens_count": 2}
{"word": " kinh▁doanh",  "impact": 5395362, "qwen_tokens": [" k", "inh", " do", "anh"], "freq": 1798454, "qwen_tokens_count": 4}
{"word": " doanh▁nghiệp","impact": 5115428, "qwen_tokens": [" do", "anh", " nghiệp"], "freq": 2557714, "qwen_tokens_count": 3}
{"word": " sử▁dụng",     "impact": 5070553, "qwen_tokens": [" sử", " dụng"], "freq": 5070553, "qwen_tokens_count": 2}
{"word": " kinh▁tế",     "impact": 4830312, "qwen_tokens": [" k", "inh", " tế"], "freq": 2415156, "qwen_tokens_count": 3}
{"word": " tổ▁chức",     "impact": 4730918, "qwen_tokens": [" tổ", " chức"], "freq": 4730918, "qwen_tokens_count": 2}
{"word": " cơ▁quan",     "impact": 4695570, "qwen_tokens": [" cơ", " qu", "an"], "freq": 2347785, "qwen_tokens_count": 3}

xzcat data/vi_words_impact.jsonl.xz | head -n 3000 |  tail -n 10 
{"word": " bàng▁quang",  "impact": 101721, "qwen_tokens": [" b", "àng", " qu", "ang"], "freq": 33907, "qwen_tokens_count": 4}
{"word": " phản▁xạ",     "impact": 101720, "qwen_tokens": [" phản", " x", "ạ"], "freq": 50860, "qwen_tokens_count": 3}
{"word": " Sức▁khỏe",    "impact": 101626, "qwen_tokens": [" S", "ức", " khỏe"], "freq": 50813, "qwen_tokens_count": 3}
{"word": " đi▁nữa",      "impact": 101591, "qwen_tokens": [" đi", " nữa"], "freq": 101591, "qwen_tokens_count": 2}
{"word": " trại▁giam",   "impact": 101586, "qwen_tokens": [" tr", "ại", " g", "iam"], "freq": 33862, "qwen_tokens_count": 4}
{"word": " giúp▁việc",   "impact": 101564, "qwen_tokens": [" giúp", " việc"], "freq": 101564, "qwen_tokens_count": 2}
{"word": " rảnh▁rỗi",    "impact": 101541, "qwen_tokens": [" r", "ảnh", " r", "ỗi"], "freq": 33847, "qwen_tokens_count": 4}
{"word": " trục▁trặc",   "impact": 101469, "qwen_tokens": [" tr", "ục", " tr", "ặc"], "freq": 33823, "qwen_tokens_count": 4}
{"word": " đau▁đầu",     "impact": 101433, "qwen_tokens": [" đau", " đầu"], "freq": 101433, "qwen_tokens_count": 2}
{"word": "Hồ▁Chí▁Minh",  "impact": 101424, "qwen_tokens": ["H", "ồ", " Chí", " Minh"], "freq": 33808, "qwen_tokens_count": 4}

xzcat data/vi_words_impact.jsonl.xz | tail -n 10
{"word": " ngân▁hà",     "impact": 6017, "qwen_tokens": [" ngân", " hà"], "freq": 6017, "qwen_tokens_count": 2}
{"word": " sông▁núi",    "impact": 6016, "qwen_tokens": [" sông", " núi"], "freq": 6016, "qwen_tokens_count": 2}
{"word": " Giới▁hạn",    "impact": 6015, "qwen_tokens": [" Giới", " hạn"], "freq": 6015, "qwen_tokens_count": 2}
{"word": " thiếu▁gia",   "impact": 6015, "qwen_tokens": [" thiếu", " gia"], "freq": 6015, "qwen_tokens_count": 2}
{"word": " Yên▁Hòa",     "impact": 6014, "qwen_tokens": [" Yên", " Hòa"], "freq": 6014, "qwen_tokens_count": 2}
{"word": " hầu▁phòng",   "impact": 6013, "qwen_tokens": [" hầu", " phòng"], "freq": 6013, "qwen_tokens_count": 2}
{"word": " kịch▁nói",    "impact": 6012, "qwen_tokens": [" kịch", " nói"], "freq": 6012, "qwen_tokens_count": 2}
{"word": " chỉ▁giới",    "impact": 6011, "qwen_tokens": [" chỉ", " giới"], "freq": 6011, "qwen_tokens_count": 2}
{"word": " ban▁ơn",      "impact": 6006, "qwen_tokens": [" ban", " ơn"], "freq": 6006, "qwen_tokens_count": 2} 
{"word": " tòa▁thành",   "impact": 6002, "qwen_tokens": [" tòa", " thành"], "freq": 6002, "qwen_tokens_count": 2}
```
![](img/vi-words-impact-01.jpg)

=> **Chọn khoảng 2k - 4k - 6k từ ghép để mở rộng vocab là hợp lý là đủ để tạo impact**

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
- Save vram while training!
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
