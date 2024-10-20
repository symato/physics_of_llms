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

x = model_path.lower()

if "qwen" in x:
    model_type = "qwen"
    from qwen_vocab import old2new, new2old, tknz_encode, tknz_decode
    STOP_WORDS = "<|im_end|> <|endoftext|>".split()

else:
    assert False, "Không hỗ trợ"


def map_tids(map_dict, tids):
    return [ map_dict[x] for x in tids if x in map_dict ]


class KeywordsStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, str):
        self.keyword_ids = tokenizer.encode(str)
        self.keyword_ids = map_tids(old2new, self.keyword_ids)
        self.keyword_len = len(self.keyword_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_ids = input_ids[0][-self.keyword_len:]
        return last_token_ids.tolist() == self.keyword_ids

stop_criteria_list = transformers.StoppingCriteriaList(
    [ KeywordsStoppingCriteria(x) for x in STOP_WORDS ]
)


def get_answer(q):
    if len(q) < 3: return "..."

    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant"
    tids = tknz_encode(prompt, tokenizer)
    input_ids = torch.tensor([ tids ]).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=1.0, top_k=30, do_sample=True,
            repetition_penalty=1.1,
            stopping_criteria=stop_criteria_list,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tids = output_ids[0][len(tids) : ].tolist() # bỏ đi prompt tokens
    return tknz_decode(answer_tids, tokenizer).split("<|im_end|>")[0].strip()


from utils import *
while True:
    # bỏ qua lỗi utf-8 encoding trong trường hợp nhập text từ console
    try: q = input(f"Bạn: {GREEN}").encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except Exception as e: print(f"{RESET}{e}"); q = ""

    reset_timer(timer=model_path)
    a = get_answer(q).strip()
    print(f"{RESET}Bot: {RED}{a}{RESET}")
    measure_time("timespent", timer=model_path)

'''
python3 model_chat.py ../Qwen2.5-1.5B-Instruct__extend_vocab

Việt Nam có gì hấp dẫn qua thời gian?

số tuổi của An trừ đi số tuổi của Lan là 3, An 10 tuổi hỏi Lan mấy tuổi?

ai tạo ra bạn

Bạn: tạo ra một câu hoàn chỉnh với từ "thực hiện"
Bot: Thì ra, việc thực hiện kế hoạch của chúng ta cần được lên lịch cụ thể.
'''
