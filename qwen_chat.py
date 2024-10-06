import torch, sys
import transformers
import config

try: model_path = sys.argv[1]
except: model_path = config.TRIMMED_MODEL_PATH

print(f"Loading {model_path} ...")

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

from qwen_vocab import old2new, new2old

def map_tids(map_dict, tids):
    if "trimm_vocab" in model_path:
        try: tids_ = tids.tolist()
        except: tids_ = tids

        for idx, x in enumerate(tids_):
            tids[idx] = map_dict[x]


STOP_WORDS = "<|im_end|> <|endoftext|>".split()
class KeywordsStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, str):
        self.keyword_ids = tokenizer(str).input_ids
        map_tids(old2new, self.keyword_ids)
        self.keyword_len = len(self.keyword_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_ids = input_ids[0][-self.keyword_len:]
        return last_token_ids.tolist() == self.keyword_ids

stop_criteria_list = transformers.StoppingCriteriaList(
    [ KeywordsStoppingCriteria(x) for x in STOP_WORDS ]
)


def get_answer(q):
    prompt = f"""<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    map_tids(old2new, inputs["input_ids"][0])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            # input_ids=input_ids,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=1.0, top_k=30, do_sample=True,
            repetition_penalty=1.1,
            stopping_criteria=stop_criteria_list,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tids = output_ids[0][len(inputs["input_ids"][0]) : ] # bỏ đi prompt tokens
    map_tids(new2old, answer_tids)

    return tokenizer.decode(answer_tids).split("<|im_end|>")[0].strip()


from utils import *
while True:
    # bỏ qua lỗi utf-8 encoding trong trường hợp nhập text từ console
    try: q = input(f"Bạn: {GREEN}").encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except Exception as e: print(e); q = ""

    reset_timer(timer=model_path)
    a = get_answer(q).strip()
    print(f"Bot: {RED}{a}{RESET}")
    measure_time("timespent", timer=model_path)


'''
python3 qwen_chat.py ../Qwen2.5-1.5B-Instruct__trimmed_vocab

python3 qwen_chat.py ../Qwen2.5-1.5B-Instruct

số tuổi của An trừ đi số tuổi của Lan là 3, An 10 tuổi hỏi Lan mấy tuổi?

ai tạo ra bạn

Bạn: tạo ra một câu hoàn chỉnh với từ "thực hiện"
Thì ra, việc thực hiện kế hoạch của chúng ta cần được lên lịch cụ thể.
'''
