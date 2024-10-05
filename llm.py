import os

'''
##########
# Ollama #
##########
import ollama
OLLAMA_CLIENT = ollama.Client(host='http://localhost:11434')
MODEL_NAME = "command-r:35b-08-2024-q8_0"
CTXLEN = 1024

def chat(prompt, temperature = 0.5, model = MODEL_NAME):
    return OLLAMA_CLIENT.chat(model = model,
        messages = [{"role": "user", "content": prompt}],
        options = {'temperature': temperature, 'num_ctx': CTXLEN,},
    )["message"]["content"].strip()
'''

'''
##########
# Gemini #
##########
PRO_MODEL = 'gemini-1.5-pro-002'
FLASH_MODEL = 'gemini-1.5-flash-002'
FLASH_8B_MODEL = 'gemini-1.5-flash-8b'

MAX_OUTPUT_TOKENS = 1024*8
TEMPERATURE = 0.3

# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Streaming.ipynb
import google.generativeai as genai # pip install -U -q google-generativeai

genai.configure(api_key = os.getenv("GEMINI_FLASH_API_KEY"))

def chat(prompt, temperature = TEMPERATURE, model = FLASH_MODEL):
    generation_config = genai.GenerationConfig(max_output_tokens = MAX_OUTPUT_TOKENS, temperature = temperature)
    genmodel = genai.GenerativeModel(model, generation_config = generation_config)

    res = genmodel.generate_content(prompt)
    return res.text
'''

'''
##############
# Openrouter #
##############
from openai import OpenAI # pip install openai
from os import getenv
import time

MODEL_NAME = "meta-llama/llama-3.1-405b-instruct:free"

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

def chat(prompt, temperature = None, model = MODEL_NAME):

    trials = MAX_TRIALS = 5
    while trials > 0:
        completion = client.chat.completions.create(
          model = model,
          messages = [{ "role": "user", "content": prompt, }]
        )
        if "rate_limit_exceeded" in str(completion):
            trials -= 1
            time.sleep(5 * (MAX_TRIALS - trials))

    try:
        return completion.choices[0].message.content
    except:
        print(completion)
        return None
'''

# '''
############
# Together #
############
from together import Together # pip install together
together_client = Together(api_key = os.getenv("TOGETHER_API_KEY"))

MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
MAX_OUTPUT_TOKENS = 1024*4

def chat(prompt, model = MODEL_NAME, temperature = 0.5):
    response = together_client.chat.completions.create(
        model = model,
        messages = [{ "role": "user", "content": prompt, }],

        max_tokens = MAX_OUTPUT_TOKENS,
        temperature = temperature,

        top_p = 0.7, top_k = 50,
        repetition_penalty = 1.2, 
    )
    return response.choices[0].message.content
# '''