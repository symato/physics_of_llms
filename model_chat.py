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
    from qwen_vocab import old2new, new2old
    STOP_WORDS = "<|im_end|> <|endoftext|>".split()

elif "gemma" in x:
    model_type = "gemma"
    from gemma_vocab import old2new, new2old
    STOP_WORDS = "<end_of_turn> <eos>".split()
else:
    assert False


def map_tids(map_dict, tids):
    if "trimm_vocab" not in model_path:
        return tids
    else:
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
    if model_type == "qwen":
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant"
    else:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False)

    old_tids = tokenizer.encode(prompt)

    new_tids = map_tids(old2new, old_tids)
    new_old_tids = map_tids(new2old, new_tids)

    new_prompt = tokenizer.decode(new_old_tids)

    if new_old_tids != old_tids:
        print(f"!!! Cảnh báo sự trimm vocab làm mất thông tin !!!")
        print(f"!!! old prompt: {prompt}")
        print(f"!!! new prompt: {new_prompt}")

    inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)

    assert inputs["input_ids"][0].tolist() == new_old_tids

    for i, x in enumerate(new_tids):
        inputs["input_ids"][0][i] = x

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=1.0, top_k=30, do_sample=True,
            repetition_penalty=1.1,
            stopping_criteria=stop_criteria_list,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tids = output_ids[0][len(inputs["input_ids"][0]) : ] # bỏ đi prompt tokens
    old_tids = map_tids(new2old, answer_tids.tolist())

    # print(prompt, answer_tids, old_tids) # DEBUG
    return tokenizer.decode(old_tids)\
        .split("<|im_end|>")[0].split("<end_of_turn>")[0].strip()


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
