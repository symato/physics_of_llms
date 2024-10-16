import torch
import transformers
import sys
import config
import re
import json

import argparse

parser = argparse.ArgumentParser(description = "LLM Model Edit, cắt tỉa embedding và mở rộng vocab")
parser.add_argument("-b", "--base_model", type = str, default = config.OFFLINE_MODEL_PATH, help = "Base model directory")
parser.add_argument("-m", "--model", type = str, default = config.OFFLINE_MODEL_PATH, help = "Model directory to apply task")
parser.add_argument("-t", "--task", type = str, default = None, \
    help = "Tác vụ `trimm_vocab` để cắt tỉa, `extend_vocab` để mở rộng")

args = parser.parse_args()
print(args)

# bỏ / ở cuối model_path
model_path = re.sub(r'/*$', "", args.model.strip())
new_model_path = f'{model_path.replace("__trimm_vocab", "")}__{args.task}'

model = transformers.AutoModelForCausalLM.from_pretrained(
   model_path,
   torch_dtype = torch.bfloat16, # dtype gốc của qwen, gemma tự động convert f32 về bf16
   device_map = "cpu"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

print(model.model)

x = model.lm_head.weight == model.model.embed_tokens.weight
is_tied_embedding = torch.all(x)


x = model_path.lower()

if "qwen" in x: 
    from qwen_vocab import kept_tids

elif "gemma" in x:
    from gemma_vocab import kept_tids

else:
    assert False


import math
n = len(kept_tids)
nn = math.ceil(n / 64) * 64


old_embeddings = model.model.embed_tokens.weight.detach().clone()
print("old_embeddings", old_embeddings.shape) # torch.Size([151936, 1536])


if is_tied_embedding:
    # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
    # embeddings chiếm 233m (~15%)
    print("tie_word_embeddings", "=> chỉ cần thay đổi model.model.embed_tokens")

    if args.task == "trimm_vocab":
        # Thay embeddings
        model.resize_token_embeddings(nn)
        new_embeddings = model.model.embed_tokens.weight.detach()
        print(new_embeddings.shape) # torch.Size([101056, 1536])

        for idx, tid in enumerate(kept_tids):
            new_embeddings[idx] = old_embeddings[tid]

        assert idx == n - 1

        embedding_size = new_embeddings.shape[1]
        assert embedding_size == 1536

        new_embeddings[n : ] = torch.zeros(nn - n, embedding_size)

        # print(new_embeddings[n : ]); input() # DEBUG, must be all 0

        x = model.model.embed_tokens.weight == new_embeddings
        assert torch.all(x), "Không thay được new_embeddings"


    elif args.task == "extend_vocab":

        vocab_size, _ = old_embeddings.shape

        if args.base_model == args.model:

            assert vocab_size == 151936
            base_embeddings = old_embeddings # same model

        else:

            assert vocab_size == 101056 # qwen 2.5 sau khi trimm vocab

            # load base embeddings
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
               args.base_model,
               torch_dtype = torch.bfloat16, # dtype gốc của qwen
               device_map = "cpu"
            )

            base_embeddings = base_model.model.embed_tokens.weight.detach().clone()

            for i in range(101012, 101056):
                x = len( torch.nonzero(old_embeddings[i]) )
                assert x == 0, f"Phần thừa sau khi làm tròn vocab size phải là 0, {old_embeddings[i]}"

        print("base_embeddings", base_embeddings.shape)

        from similarity import get_similiar_words
        words = get_similiar_words()
        added_tokens_count = len(words)

        # Làm tròn lên hệ số 64
        if added_tokens_count % 64 != 0:
            added_tokens_count += (64 - added_tokens_count % 64)

        # print(words)
        print(f"Adding {added_tokens_count} new tokens ...")

        model.resize_token_embeddings(vocab_size + added_tokens_count)
        new_embeddings = model.model.embed_tokens.weight.detach()

        redudant = added_tokens_count - len(words)
        new_embeddings[ -redudant-1 : ] = torch.zeros(redudant+1, new_embeddings.shape[1])
        print(new_embeddings[ -redudant : ]) # DEBUG

        word2tid = {}
        for idx, (k, v) in enumerate(words):
            english_tids = v.values()

            embeddings_ = []
            for tid in english_tids:
                embeddings_.append( base_embeddings[tid].tolist() )
            
            new_tid = vocab_size + idx
            word2tid[k] = new_tid

            # print(f"Tạo embedding value cho new token #{new_tid} {k}")

            embeddings_avg = torch.Tensor(embeddings_).mean(dim=0, keepdim=True)
            new_embeddings[ new_tid ] = embeddings_avg

        print(new_embeddings[ -redudant - 1 : ])#; input() # DEBUG

        x = model.model.embed_tokens.weight == new_embeddings
        assert torch.all(x), "Không thay được new_embeddings"

        filename = f"data/new_words.json"
        print(f"{len(word2tid)} words add. See {filename}")
        with open(filename, "wt") as f:
            f.write(json.dumps(word2tid, ensure_ascii = False))


    if args.task == "view_embeddings":
        vocab_size, embed_size = old_embeddings.shape
        assert vocab_size == 102080
        assert embed_size == 1536

        # 101011 - 101055 được gán 0
        # 102075 - 102079 được gán 0 too.

        for i in range(101011, 101056):
            print(old_embeddings[i])
            x = len( torch.nonzero(old_embeddings[i]) )
            assert x == 0, f"Phần thừa sau khi làm tròn vocab size phải là 0, {old_embeddings[i]}"

    else:
        assert False, "Không hỗ trợ task này" 


    print("model.model.embed_tokens.weight", model.model.embed_tokens.weight.shape)

else:
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
    # embeddings chiếm 1b (~15%)
    print("separate embeddings", "=> cần tỉa cả embed_tokens và lm_head")

    if args.task == "trimm_vocab":

        old_lm_head = model.lm_head.weight.detach().clone()
        print("old_lm_head", old_lm_head.shape) # torch.Size([152064, 3584])

        # Thay embeddings
        model.resize_token_embeddings(nn)
        new_embeddings = model.model.embed_tokens.weight.detach()

        new_lm_head = model.lm_head.weight.detach()
        print(new_lm_head.shape) # torch.Size([101056, 3584])

        for idx, tid in enumerate(kept_tids):
            new_embeddings[idx] = old_embeddings[tid]
            new_lm_head[idx] = old_lm_head[tid]

        x = model.model.embed_tokens.weight == new_embeddings
        assert torch.all(x), "Không thay được new_embeddings"

        x = model.lm_head.weight == new_lm_head
        assert torch.all(x), "Không thay được new_lm_head"


    elif args.task == "extend_vocab":

        vocab_size, _ = old_embeddings.shape

        if args.base_model == args.model:

            assert vocab_size == 151936
            base_embeddings = old_embeddings # same model

        else:

            assert vocab_size == 101056 # qwen 2.5 sau khi trimm vocab

            # load base embeddings
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
               args.base_model,
               torch_dtype = torch.bfloat16, # dtype gốc của qwen
               device_map = "cpu"
            )

            base_embeddings = base_model.model.embed_tokens.weight.detach().clone()
            base_lm_head    = base_model.lm_head.weight.detach().clone()

        print("base_embeddings", base_embeddings.shape)

        from similarity import get_similiar_words
        words = get_similiar_words()
        added_tokens_count = len(words)

        # Làm tròn lên hệ số 64
        if added_tokens_count % 64 != 0:
            added_tokens_count += (64 - added_tokens_count % 64)

        # print(words)
        print(f"Adding {added_tokens_count} new tokens ...")

        model.resize_token_embeddings(vocab_size + added_tokens_count)
        new_embeddings = model.model.embed_tokens.weight.detach()

        new_lm_head = model.lm_head.weight.detach()
        print(new_lm_head.shape)

        for idx, (k, v) in enumerate(words):
            english_tids = v.values()

            embeddings_ = []
            lm_head_ = []
            for tid in english_tids:
                embeddings_.append( base_embeddings[tid].tolist() )
                lm_head_.append( base_lm_head[tid].tolist() )
            
            new_tid = vocab_size + idx

            # print(f"Tạo embedding value cho new token #{new_tid} {k}")

            embeddings_avg = torch.Tensor(embeddings_).mean(dim=0, keepdim=True)
            new_embeddings[ new_tid ] = embeddings_avg

            lm_head_avg = torch.Tensor(lm_head_).mean(dim=0, keepdim=True)
            new_lm_head[ new_tid ] = lm_head_avg

        x = model.model.embed_tokens.weight == new_embeddings
        assert torch.all(x), "Không thay được new_embeddings"

        x = model.lm_head.weight == new_lm_head
        assert torch.all(x), "Không thay được new_lm_head"

    else:
        assert False, "Không hỗ trợ task này" 

model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)


'''

python model_edit.py -m ../gemma-2-2b-it/ -t trimm_vocab

python model_edit.py -m ../gemma-2-2b-it__trimm_vocab/ -t extend_vocab


python model_edit.py -m ../Qwen2.5-1.5B-Instruct/ -t trimm_vocab

python model_edit.py -m ../Qwen2.5-1.5B-Instruct__trimm_vocab/ -t extend_vocab


Gemma2 2b gồm tied embeddings và 26 layers

Gemma2Model(
  (embed_tokens): Embedding(256000, 2304, padding_idx=0)
  (layers): ModuleList(
    (0-25): 26 x Gemma2DecoderLayer(
      (self_attn): Gemma2SdpaAttention(
        (q_proj): Linear(in_features=2304, out_features=2048, bias=False)
        (k_proj): Linear(in_features=2304, out_features=1024, bias=False)
        (v_proj): Linear(in_features=2304, out_features=1024, bias=False)
        (o_proj): Linear(in_features=2048, out_features=2304, bias=False)
        (rotary_emb): Gemma2RotaryEmbedding()
      )
      (mlp): Gemma2MLP(
        (gate_proj): Linear(in_features=2304, out_features=9216, bias=False)
        (up_proj): Linear(in_features=2304, out_features=9216, bias=False)
        (down_proj): Linear(in_features=9216, out_features=2304, bias=False)
        (act_fn): PytorchGELUTanh()
      )
      (input_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
      (post_attention_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
      (pre_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
      (post_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
    )
  )
  (norm): Gemma2RMSNorm((2304,), eps=1e-06)
)


Qwen's 1.5b gồm 28 layers, với tied embeddings là embed_tokens.weight (model.norm.weight là RMS Norm)

Qwen2Model(
  (embed_tokens): Embedding(151936, 1536)
  (layers): ModuleList(
    (0-27): 28 x Qwen2DecoderLayer(
      (self_attn): Qwen2SdpaAttention(
        (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
        (k_proj): Linear(in_features=1536, out_features=256, bias=True)
        (v_proj): Linear(in_features=1536, out_features=256, bias=True)
        (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (mlp): Qwen2MLP(
        (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
        (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
        (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
    )
  )
  (norm): Qwen2RMSNorm((1536,), eps=1e-06)
  (rotary_emb): Qwen2RotaryEmbedding()
)

'''
