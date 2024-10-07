import os

model = os.getenv("model", "1.5b")

if model   == "0.5b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-0.5B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-0.5B-Instruct__trimm_vocab"

elif model == "1b":

	ONLINE_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
	OFFLINE_MODEL_PATH = "../Llama-3.2-1B-Instruct"
	TRIMMED_MODEL_PATH = "../Llama-3.2-1B-Instruct__trimm_vocab"

elif model == "1.5b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-1.5B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-1.5B-Instruct__trimm_vocab"

elif model == "2b":

	ONLINE_MODEL_PATH = "google/gemma-2-2b-it"
	OFFLINE_MODEL_PATH = "../gemma-2-2b-it"
	TRIMMED_MODEL_PATH = "../gemma-2-2b-it__trimm_vocab"

elif model == "3b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-3B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-3B-Instruct__trimm_vocab"

elif model == "3.2b":

	ONLINE_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
	OFFLINE_MODEL_PATH = "../Llama-3.2-3B-Instruct"
	TRIMMED_MODEL_PATH = "../Llama-3.2-3B-Instruct__trimm_vocab"

elif model == "7b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-7B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-7B-Instruct__trimm_vocab"

elif model == "8b":

	ONLINE_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
	OFFLINE_MODEL_PATH = "../Llama-3.1-8B-Instruct"
	TRIMMED_MODEL_PATH = "../Llama-3.1-8B-Instruct__trimm_vocab"

elif model == "9b":

	ONLINE_MODEL_PATH = "google/gemma-2-9b-it"
	OFFLINE_MODEL_PATH = "../gemma-2-9b-it"
	TRIMMED_MODEL_PATH = "../gemma-2-9b-it__trimm_vocab"

elif model == "14b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-14B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-14B-Instruct__trimm_vocab"

elif model == "27b":

	ONLINE_MODEL_PATH = "google/gemma-2-27b-it"
	OFFLINE_MODEL_PATH = "../gemma-2-27b-it"
	TRIMMED_MODEL_PATH = "../gemma-2-27b-it__trimm_vocab"

elif model == "32b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-32B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-32B-Instruct__trimm_vocab"

elif model == "70b":

	ONLINE_MODEL_PATH = "meta-llama/Llama-3.1-70B-Instruct"
	OFFLINE_MODEL_PATH = "../Llama-3.1-70B-Instruct"
	TRIMMED_MODEL_PATH = "../Llama-3.1-70B-Instruct__trimm_vocab"

elif model == "72b":

	ONLINE_MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
	OFFLINE_MODEL_PATH = "../Qwen2.5-72B-Instruct"
	TRIMMED_MODEL_PATH = "../Qwen2.5-72B-Instruct__trimm_vocab"

elif model == "405b":

	ONLINE_MODEL_PATH = "meta-llama/Llama-3.1-405B-Instruct"
	OFFLINE_MODEL_PATH = "../Llama-3.1-405B-Instruct"
	TRIMMED_MODEL_PATH = "../Llama-3.1-405B-Instruct__trimm_vocab"

else:
	assert False, "Chưa hỗ trợ"

'''

if   model == "0.5b":
elif model == "1b":
elif model == "1.5b":
elif model == "2b":
elif model == "3b":
elif model == "3.2b":
elif model == "7b":
elif model == "8b":
elif model == "9b":
elif model == "14b":
elif model == "27b":
elif model == "32b":
elif model == "70b":
elif model == "72b":
elif model == "405b":

'''
