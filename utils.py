import time, os
import subprocess
import re
import fasttext # pip install fasttext

LOCATION = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

## Các màu hay dùng
BLACK   = '\033[30m';   WHITE  = '\033[97m'
RED     = '\033[91m';   YELLOW = '\033[33m'
GREEN   = '\033[32m';   CYAN   = '\033[36m'
BLUE    = '\033[94m';   GREY   = '\033[37m'
MAGENTA = '\033[95m';   RESET  = '\033[0m'


def num_procs():
    return os.cpu_count() - 2

TIMER_STARTED_AT = { "default": time.time() }
def reset_timer(timer="default"):
    global TIMER_STARTED_AT
    TIMER_STARTED_AT[timer] = time.time()


def measure_time(message="", timer="default", color=YELLOW):
    total = time.time() - TIMER_STARTED_AT[timer]
    total = pretty_num(total)

    message = message.strip()
    if len(message) > 0: 
        message = " " + message

    print(f"{color}{timer}:{message} {total} seconds{RESET}")


def count_words(x):
    return len(x.split())

def pretty_num(x):
    return round(x*100)/100

def mkdirs(path):
    splits = path.split("/")
    for i in range(0, len(splits)):
        x = "/".join(splits[ : i + 1])
        cmd = f"mkdir -p {x}"
        print(cmd)
        subprocess.run(cmd, shell = True)


## Fasttext detect lang
LANGID_FILENAME = 'lid.176.bin'
LANGID_FILEPATH = f"{LOCATION}/data/{LANGID_FILENAME}"

if not os.path.exists(LANGID_FILEPATH):
    cmd = f"wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/{LANGID_FILENAME}; mv {LANGID_MODEL} {LANGID_FILEPATH}"
    subprocess.run(cmd, shell=True)

FASTTEXT_MODEL = fasttext.load_model(LANGID_FILEPATH)
WORD_RE = re.compile(r'\w+\s')

def detect_lang(text, check_words = False):
    if check_words:
        # Chỉ kiểm tra ngôn ngữ với những từ được phân tách rõ ràng
        words = re.findall(WORD_RE, text)
        text = " ".join(words)

    rs = FASTTEXT_MODEL.f.predict(text, 1, 0.0, 'strict')
    if rs: return rs[0][-1].split('__')[-1]
    else:  return None

assert detect_lang("hello vietnam") == "en"
assert detect_lang( "chào nước mỹ 123sfd http://adf4| tôi là ") == "vi"


if __name__ == "__main__":

    reset_timer(timer="my timer")

    s = "chào cả nhà, cả nhà khỏe không ạ?"
    print(f"{RED}{s}{RESET} có {CYAN}{count_words(s)} từ")

    measure_time("tổng thời gian chạy", timer="my timer")
