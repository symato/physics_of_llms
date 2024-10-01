import time, os
location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

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


if __name__ == "__main__":
	reset_timer(timer="my timer")

	s = "chào cả nhà, cả nhà khỏe không ạ?"
	print(f"{RED}{s}{RESET} có {CYAN}{count_words(s)} từ")

	measure_time("tổng thời gian chạy", timer="my timer")
