prompt_template = """
INPUT
"miễn cưỡng" dịch sang tiếng Anh là gì? 
Cho tôi khoảng 3 lựa chọn và với mỗi lựa chọn hãy giải thích tại sao nó phù hợp và cho ví dụ bằng tiếng Anh và tiếng Việt

OUPUT
Từ "miễn cưỡng" trong tiếng Việt có thể được dịch sang tiếng Anh với nhiều sắc thái khác nhau tùy thuộc vào ngữ cảnh. Dưới đây là 3 lựa chọn phổ biến:

1. **Reluctant**  
   - Giải thích: "Reluctant" là lựa chọn phổ biến nhất khi nói về sự miễn cưỡng, ngụ ý ai đó không muốn làm điều gì đó nhưng vẫn phải làm. Từ này diễn tả cảm giác thiếu hứng thú hoặc sự dè dặt.
   - Ví dụ:  
     - He was reluctant to attend the meeting.  
     - Anh ấy miễn cưỡng tham dự cuộc họp.

2. **Unwilling**  
   - Giải thích: "Unwilling" nhấn mạnh hơn về việc không sẵn lòng hoặc không muốn làm điều gì đó. So với "reluctant", "unwilling" có thể mang ý nghĩa mạnh hơn, gần với việc từ chối.
   - Ví dụ:  
     - She was unwilling to accept the offer.  
     - Cô ấy miễn cưỡng chấp nhận lời đề nghị.

3. **Hesitant**  
   - Giải thích: "Hesitant" miêu tả sự do dự hoặc không chắc chắn, nhưng vẫn có ý định có thể làm điều đó. Từ này có thể sử dụng khi ai đó miễn cưỡng nhưng còn lưỡng lự và chưa quyết định rõ ràng.
   - Ví dụ:  
     - He was hesitant to speak up during the discussion.  
     - Anh ấy miễn cưỡng phát biểu trong cuộc thảo luận.

INPUT
"{word}" dịch sang tiếng Anh là gì? 
Cho tôi khoảng 3 lựa chọn và với mỗi lựa chọn hãy giải thích tại sao nó phù hợp và cho ví dụ bằng tiếng Anh và tiếng Việt

OUPUT
""".strip()

import ollama
OLLAMA_CLIENT = ollama.Client(host='http://localhost:11434')
OLLAMA_MODEL = "llama3.1:70b-instruct-q3_K_L"
CTXLEN = 1024

def chat(prompt, temperature = 0.5):
    return OLLAMA_CLIENT.chat(model = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}],
        options = {'temperature': temperature, 'num_ctx': CTXLEN,},
    )["message"]["content"].strip()


import lzma, json, os
from utils import *

infile = "data/vi_words_score.jsonl.xz"
outfile = "data/vi_words_similarity.jsonl.xz"

try:
    done = [ json.loads(line)["source"] for line in lzma.open(outfile) ]
except:
    done = []

for idx, line in enumerate( lzma.open(infile) ):
    # Thử trước với 100 words
    if idx >= 100: break

    source = f"{infile}:{idx}"

    if source not in done:

        word = json.loads(line)["word"].replace("_", " ")

        prompt = prompt_template.format( word = word )

        print(source)

        reset_timer()
        res = chat(prompt)

        print(res)
        measure_time(OLLAMA_MODEL)

        with lzma.open(outfile, "at") as f:
            f.write(json.dumps({
                "word": word,
                "textbook": res,
                "source": source,
            }, ensure_ascii = False) + "\n")
