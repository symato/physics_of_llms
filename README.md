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

python3 qwen_edit.py -m ../Qwen2.5-1.5B-Instruct -t trimm_vocab

python3 qwen_chat.py ../Qwen2.5-1.5B-Instruct__trimm_vocab
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


## Physics of LMs: Sau khi tỉa gọn, từng bước một mở rộng vocab

- [x] tạo bộ từ điển từ ghép tiếng Việt thông dụng và chiếm nhiều tokens nhất, chỉ cần khoảng < 10k
  - Dùng https://github.com/trungtv/pyvi để tách từ
```sh
# `impact = freq * (qwen_tokens_count - 1)` (`freq` là tần suất sử dụng của từ đó trong corpus)
xzcat data/vi_words_impact.jsonl.xz | head -n 10
{"word": " có_thể",       "impact": 3155411, "qwen_tokens": [" có", " thể"], "freq": 3155411, "qwen_tokens_count": 2}
{"word": " thời_gian",    "impact": 2239268, "qwen_tokens": [" thời", " g", "ian"], "freq": 1119634, "qwen_tokens_count": 3}      
{"word": " Việt_Nam",     "impact": 1969804, "qwen_tokens": [" Việt", " Nam"], "freq": 1969804, "qwen_tokens_count": 2}
{"word": " thực_hiện",    "impact": 1417257, "qwen_tokens": [" thực", " hiện"], "freq": 1417257, "qwen_tokens_count": 2}
{"word": " kinh_doanh",   "impact": 1363467, "qwen_tokens": [" k", "inh", " do", "anh"], "freq": 454489, "qwen_tokens_count": 4}  
{"word": " doanh_nghiệp", "impact": 1358574, "qwen_tokens": [" do", "anh", " nghiệp"], "freq": 679287, "qwen_tokens_count": 3}    
{"word": " sử_dụng",      "impact": 1331852, "qwen_tokens": [" sử", " dụng"], "freq": 1331852, "qwen_tokens_count": 2}
{"word": " kinh_tế",      "impact": 1255572, "qwen_tokens": [" k", "inh", " tế"], "freq": 627786, "qwen_tokens_count": 3}
{"word": " tổ_chức",      "impact": 1253207, "qwen_tokens": [" tổ", " chức"], "freq": 1253207, "qwen_tokens_count": 2}
{"word": " cơ_quan",      "impact": 1217440, "qwen_tokens": [" cơ", " qu", "an"], "freq": 608720, "qwen_tokens_count": 3}

xzcat data/vi_words_impact.jsonl.xz | head -n 3000 |  tail -n 10 
{"word": " phong_bì",     "impact": 23052, "qwen_tokens": [" ph", "ong", " b", "ì"], "freq": 7684, "qwen_tokens_count": 4}
{"word": " Anh_em",       "impact": 23048, "qwen_tokens": [" An", "h", " em"], "freq": 11524, "qwen_tokens_count": 3}
{"word": " lật_đổ",       "impact": 23034, "qwen_tokens": [" l", "ật", " đổ"], "freq": 11517, "qwen_tokens_count": 3}
{"word": " sữa_chua",     "impact": 23032, "qwen_tokens": [" sữa", " ch", "ua"], "freq": 11516, "qwen_tokens_count": 3}
{"word": " vững_mạnh",    "impact": 23012, "qwen_tokens": [" vững", " mạnh"], "freq": 23012, "qwen_tokens_count": 2}
{"word": " thủ_công",     "impact": 23009, "qwen_tokens": [" thủ", " công"], "freq": 23009, "qwen_tokens_count": 2}
{"word": " Điện_thoại",   "impact": 23000, "qwen_tokens": [" Điện", " thoại"], "freq": 23000, "qwen_tokens_count": 2}
{"word": " Trường_Sa",    "impact": 22989, "qwen_tokens": [" Trường", " Sa"], "freq": 22989, "qwen_tokens_count": 2}
{"word": " hiển_nhiên",   "impact": 22986, "qwen_tokens": [" hiển", " nhiên"], "freq": 22986, "qwen_tokens_count": 2}
{"word": " Kỹ_năng",      "impact": 22982, "qwen_tokens": [" K", "ỹ", " năng"], "freq": 11491, "qwen_tokens_count": 3}

xzcat data/vi_words_impact.jsonl.xz | tail -n 10
{"word": " tự_phụ",       "impact": 601, "qwen_tokens": [" tự", " phụ"], "freq": 601, "qwen_tokens_count": 2}
{"word": " thành_tố",     "impact": 601, "qwen_tokens": [" thành", " tố"], "freq": 601, "qwen_tokens_count": 2}
{"word": " viên_ngoại",   "impact": 601, "qwen_tokens": [" viên", " ngoại"], "freq": 601, "qwen_tokens_count": 2}
{"word": " thánh_địa",    "impact": 601, "qwen_tokens": [" thánh", " địa"], "freq": 601, "qwen_tokens_count": 2}
{"word": " mấy_khi",      "impact": 600, "qwen_tokens": [" mấy", " khi"], "freq": 600, "qwen_tokens_count": 2}
{"word": " đậu_đen",      "impact": 600, "qwen_tokens": [" đậu", " đen"], "freq": 600, "qwen_tokens_count": 2}
{"word": " tái_bản",      "impact": 600, "qwen_tokens": [" tái", " bản"], "freq": 600, "qwen_tokens_count": 2}
{"word": " kéo_lại",      "impact": 600, "qwen_tokens": [" kéo", " lại"], "freq": 600, "qwen_tokens_count": 2}
{"word": " e_sợ",         "impact": 600, "qwen_tokens": [" e", " sợ"], "freq": 600, "qwen_tokens_count": 2}
{"word": " hồi_đáp",      "impact": 600, "qwen_tokens": [" hồi", " đáp"], "freq": 600, "qwen_tokens_count": 2}
```
![](img/vi-words-impact-00.jpg)

=> **Chọn khoảng 2.4k - 3.2k từ ghép để mở rộng vocab là hợp lý là đủ để tạo impact**

- [ ] lọc và map những từ ghép này vào token ids mới

- [ ] tìm các cách *hiệu quả* để khởi tạo embedding values của tokens mới
  - Với 1 từ được chọn, tìm ra 1-3 câu liên quan tới từ đó:
    - thay toàn bộ embedding values của từ được chọn băng 01 embedding value mới được init bằng nhiều cách:
      - embedding value của từ đơn tương ứng trong tiếng Anh hoặc Trung
      - lấy trung bình cộng của các embedding values của từ đó
      - lấy trung bình cộng của toàn bộ embedding values (whole vocab)
      - lấy trung bình cộng của toàn bộ embedding values của các từ trong ngôn ngữ đó
      - lấy trung bình cộng của toàn bộ embedding values của các từ gần nghĩa với nó trong tiếng Việt
      - lấy trung bình cộng của toàn bộ embedding values của các từ gần nghĩa với nó trong TV và các ngôn ngữ khác
      - ...
  - làm thế nào để đo lường được *hiệu quả*?
    - tính sự khác biệt của output (logits diff / perpelexity ...) trong các phép thay thế,
    khác biệt thấp nhất => hiệu quả nhất?

- [ ] Mát xa new embeddings (and old embeddings too)
  - freeze all layers, finetune embeddings trước
  - sau đó finetune models (lora + embedding or full finetune)
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

![](img/vocab-extend-03.jpg)
https://dict.laban.vn/find?type=2&query=s%E1%BB%AD+d%E1%BB%A5ng

- - -

## Physics of LMs: làm thí nghiệm về Knowledge Storage, Extraction and Manipulation

## Physics of LMs: làm thí nghiệm [TinyStories](TinyStories.md) về học language
- Build dataset theo một hướng khác? TinyFantasy? TinyFunny?
- Mở rộng: xây bộ data để chuyển knowledge đã học từ Vi => En
- Mở rộng: Từ hiểu ngôn ngữ tới làm thơ và làm thơ thuận nghịch độc
  - 750 bài thơ thuận nghịch độc https://vantholieuxuyencom.blog/about/
  - Nên bắt đầu với dataset thuận nghịch độc, các câu ngắn đọc xuôi hay ngược đều có ý nghĩa
- Ý tưởng: dùng kỹ thuật FIM (fill in middle) của code LLM để tạo thơ có vần
  Yêu cầu chữ cuối phải rơi vào 1 từ hoặc 1 vần nào đó ...

![](img/tho-thuan-nghich-doc.jpg)
