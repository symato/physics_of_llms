prompt_template = """

<INSTRUCTION>
Cho một từ tiếng Việt.
Sinh ra 5 câu sử dụng từ đó.

Tiếp theo,
Dịch từ được cho từ tiếng Việt sang tiếng Anh.
Cho tôi khoảng 3 lựa chọn và với mỗi lựa chọn hãy giải thích tại sao nó phù hợp và cho ví dụ bằng tiếng Anh và tiếng Việt
Ví dụ tiếng Việt phải dùng chính xác từ được cung cấp, không được dùng biến thể hay từ đồng nghĩa.
Cuối cùng hãy đưa ra bất kỳ nhận xét gì bạn muốn.
Nếu từ được cung cấp trùng với ví dụ thì sử dụng luôn nội dung trong ví dụ.
Sử dụng format ở các ví dụ dưới.

Ví dụ 1:
<word>sử dụng</word>
<RESPONSE>
- Chúng ta có thể đi picnic vào cuối tuần này nếu trời đẹp.
- Bạn có thể giúp tôi mang những túi đồ này lên tầng không?
- Việc học ngoại ngữ có thể mở ra nhiều cơ hội mới trong sự nghiệp.
- Nếu cố gắng, em có thể đạt được mục tiêu của mình.
- Chúng tôi có thể gặp nhau vào buổi tối để thảo luận về dự án.

Từ tiếng Việt "sử dụng" có thể được dịch sang tiếng Anh với các lựa chọn sau:

1. "Use"
   - Phù hợp vì: Đây là từ phổ biến nhất và có nghĩa tương đương trực tiếp với "sử dụng".
   - Ví dụ tiếng Anh: We should use renewable energy sources.
   - Ví dụ tiếng Việt: Chúng ta nên sử dụng các nguồn năng lượng tái tạo.

2. "Utilize"
   - Phù hợp vì: Mang nghĩa tương tự "use" nhưng thường được sử dụng trong ngữ cảnh trang trọng hơn hoặc mang tính kỹ thuật.
   - Ví dụ tiếng Anh: The company utilizes advanced technology to improve productivity.
   - Ví dụ tiếng Việt: Công ty sử dụng công nghệ tiên tiến để nâng cao năng suất.

3. "Employ"
   - Phù hợp vì: Cũng có nghĩa là "sử dụng" nhưng thường được dùng trong ngữ cảnh chuyên nghiệp hoặc học thuật.
   - Ví dụ tiếng Anh: Researchers employ various methods to collect data.
   - Ví dụ tiếng Việt: Các nhà nghiên cứu sử dụng nhiều phương pháp khác nhau để thu thập dữ liệu.

Nhận xét: 
Từ "sử dụng" trong tiếng Việt khá linh hoạt và có thể được dùng trong nhiều ngữ cảnh khác nhau, từ đời sống hàng ngày đến các tình huống chuyên môn. Khi dịch sang tiếng Anh, việc chọn từ phù hợp phụ thuộc vào ngữ cảnh cụ thể và mức độ trang trọng của văn bản. Trong khi "use" là lựa chọn phổ biến nhất và phù hợp với hầu hết các tình huống, "utilize" và "employ" có thể được ưu tiên trong các văn bản học thuật hoặc chuyên nghiệp để tạo sự đa dạng và chính xác trong cách diễn đạt.

Ví dụ 2:
<word>có thể</word>
<RESPONSE>
- Ngày mai trời có thể mưa, bạn nên mang theo ô.
- Nếu cố gắng, em có thể đạt được mục tiêu của mình.
- Chúng ta có thể gặp nhau vào cuối tuần này để thảo luận về dự án.
- Chúng ta có thể thay đổi thế giới bằng những hành động nhỏ mỗi ngày.
- Bạn có thể đạt được mọi điều nếu bạn tin tưởng vào bản thân.

Từ tiếng Việt "có thể" có thể được dịch sang tiếng Anh với các lựa chọn sau:

1. "Can"
   - Phù hợp vì: Đây là từ phổ biến nhất để diễn đạt khả năng hoặc sự cho phép.
   - Ví dụ tiếng Anh: We can change the world with small actions every day.
   - Ví dụ tiếng Việt: Chúng ta có thể thay đổi thế giới bằng những hành động nhỏ mỗi ngày.

2. "May"
   - Phù hợp vì: Thường được sử dụng để diễn đạt khả năng xảy ra hoặc sự không chắc chắn.
   - Ví dụ tiếng Anh: It may rain today, so take an umbrella when you go out.
   - Ví dụ tiếng Việt: Hôm nay trời có thể mưa, nên hãy mang theo ô khi ra ngoài.

3. "Be able to"
   - Phù hợp vì: Diễn đạt khả năng thực hiện một hành động, thường được sử dụng trong các câu phức tạp hơn.
   - Ví dụ tiếng Anh: You will be able to achieve anything if you believe in yourself.
   - Ví dụ tiếng Việt: Bạn có thể đạt được mọi điều nếu bạn tin tưởng vào bản thân.

Nhận xét:
"Có thể" trong tiếng Việt là một cụm từ đa năng, có thể được sử dụng để diễn đạt khả năng, sự cho phép, hoặc xác suất xảy ra. Khi dịch sang tiếng Anh, việc chọn từ thích hợp phụ thuộc vào ngữ cảnh và ý nghĩa cụ thể mà người nói muốn truyền đạt. "Can" thường được sử dụng nhiều nhất và phù hợp trong hầu hết các tình huống. "May" thích hợp khi nói về khả năng xảy ra của một sự kiện, đặc biệt là khi có sự không chắc chắn. "Be able to" thường được sử dụng để nhấn mạnh khả năng thực hiện một hành động, đặc biệt trong các câu phức tạp hoặc khi nói về khả năng trong tương lai.

Ví dụ 3:
<word>Hà Nội</word>
<RESPONSE>
- Tôi đã từng sống ở Hà Nội trước khi chuyển xuống thành phố Hồ Chí Minh.
- Mỗi mùa xuân, tôi lại nhớ về hương hoa gạo rực rỡ trên đường phố Hà Nội.
- Năm ngoái, chúng tôi đã tổ chức một chuyến du lịch tới thăm quan di tích lịch sử ở Hà Nội.
- Món ăn nổi tiếng nhất ở Hà Nội có lẽ là phở gà.
- Tôi yêu thích khí hậu mát mẻ và dễ chịu của mùa thu ở Hà Nội.

Từ tiếng Việt "Hà Nội" có thể được dịch sang tiếng Anh với các lựa chọn sau:

1. "Hanoi"
 - Phù hợp vì: Đây là tên gọi thông dụng và chính thức của thủ đô Việt Nam trong tiếng Anh.
 - Ví dụ tiếng Anh: I visited Hanoi last year and enjoyed its beautiful scenery.
 - Ví dụ tiếng Việt: Năm ngoái tôi đã tới thăm Hà Nội và tận hưởng phong cảnh tuyệt đẹp.

Không cần thêm lựa chọn nào khác vì đây là địa danh.
Tuy nhiên, cũng cần lưu ý rằng trong quá khứ, thành phố này còn được gọi là Thăng Long (trước năm 1831) và Đông Kinh.     

Nhận xét:
"Hà Nội" là thủ đô của Việt Nam và là thành phố lớn thứ hai cả nước. Tên gọi này gắn liền với lịch sử lâu dài và phong phú của thành phố. Khi dịch sang tiếng Anh, chỉ đơn giản là giữ nguyên tên gọi thông dụng quốc tế "Hanoi".

Tới lượt bạn
<word>{word}</word>
<RESPONSE>
""".strip()


import lzma, json, os
from utils import *
import time
import llm

infile = "data/vi_words_score.jsonl.xz"
outfile = "data/vi_words_similarity.jsonl"
model = llm.MODEL_NAME

try:
    done = [ json.loads(line)["source"] for line in open(outfile) ]
except:
    done = []

for idx, line in enumerate( lzma.open(infile) ):
    # Thử trước với 100 words
    if idx >= 128: break

    source = f"{infile}:{idx}"

    if source not in done:

        word = json.loads(line)["word"].replace("_", " ")

        prompt = prompt_template.format( word = word )

        print(source)
        reset_timer()

        res = llm.chat(prompt, model = model, temperature = 0.5)

        if res is not None:

            for x in "<INSTRUCTION> <RESPONSE>".split():
                res = res.split(x)[0].strip()

            header = "# **Dịch từ "

            if header in res:
                res = res.split(header)[-1]
                res = "# **Dịch từ " + res

            print(res)
            measure_time(model)

            assert "Ví dụ tiếng Việt:" in res, "LLM sinh thiếu ví dụ ..."
            ww = word.lower()

            for example in re.findall(r'Ví dụ(?:\n|.)+?\n\n', res, flags = re.MULTILINE | re.IGNORECASE):
                # print(example); print("!!!!!"); input() # DEBUG
                example = example.strip()
                if ww not in example.lower():
                    print(f"LLM đưa ví dụ không chuẩn cho từ '{word}' <= {example}")
                    # assert False, example


            with open(outfile, "at") as f:
                f.write(json.dumps({
                    "word": word.strip(),
                    "textbook": res.strip(),
                    "source": source.strip(),
                }, ensure_ascii = False) + "\n")

