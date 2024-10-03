import torch
import transformers
# from unsloth import FastLanguageModel

new_mode_path = "../Qwen2.5-0.5B-Instruct__trimmed_vocab"
# new_mode_path = "../Qwen2.5-0.5B-Instruct"

model = transformers.AutoModelForCausalLM.from_pretrained(
    new_mode_path, 
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(new_mode_path)

from qwen_vocab import get_kept_tids
kept_tids = get_kept_tids()
kept_tids.sort()

# old vs new vocab mapping
old2new = {}
new2old = {}
for new_tid, old_tid in enumerate( kept_tids ):
    old2new[ old_tid ] = new_tid
    new2old[ new_tid ] = old_tid


def map_tids(map_dict, tids):
    for idx, x in enumerate(tids.tolist()):
        # print(tids, idx, x) # DEBUF
        tids[idx] = map_dict[x]


STOP_WORDS = "<|im_end|> </s> <|endoftext|>".split()
class KeywordsStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, str):
        self.keyword_ids = tokenizer(str).input_ids
        self.keyword_ids = self.keyword_ids[1:]
        # map_tids(old2new, self.keyword_ids)
        self.keyword_len = len(self.keyword_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_ids = input_ids[0][-self.keyword_len:]
        return last_token_ids.tolist() == self.keyword_ids

x = [KeywordsStoppingCriteria(xx) for xx in STOP_WORDS]
stop_criteria_list = transformers.StoppingCriteriaList(x)


def get_answer(q):
    prompt = f"""<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant"""

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
    map_tids(old2new, input_ids[0])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            temperature=0.5,
            top_p=1.0, top_k=30, do_sample=True,
            repetition_penalty=1.1,
            stopping_criteria=stop_criteria_list,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tids = output_ids[0][len(input_ids[0]) : ] # bỏ đi prompt tokens
    map_tids(new2old, answer_tids)

    return tokenizer.decode(answer_tids).split("<|im_end|>")[0]


from utils import *
while True:
    # bỏ qua lỗi utf-8 encoding trong trường hợp nhập text từ console
    try: q = input(f"Bạn: {GREEN}").encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except Exception as e: print(e); q = ""

    reset_timer()
    a = get_answer(q)
    print(f"{RED}{a}{RESET}", end="")
    measure_time("")
