import torch
import transformers

model_path = "../Qwen2.5-1.5B-Instruct"
new_mode_path = "../Qwen2.5-1.5B-Instruct__trimmed_vocab"

if True:
    model = transformers.AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype = torch.bfloat16, # dtype gốc của qwen
       device_map = "cpu"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    print("lm_head", model.lm_head) # Linear(in_features=1536, out_features=151936, bias=False)
    print("embed_tokens", model.model.embed_tokens) # Embedding(151936, 1536) ~= 233m params

    x = model.lm_head.weight == model.model.embed_tokens.weight
    is_tied_embedding = torch.all(x)


    from qwen_vocab import get_kept_tids
    kept_tids = get_kept_tids()
    kept_tids.sort()

    n = len(kept_tids)
    nn = round(n / 64) * 64

    if is_tied_embedding:
       # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
       # embeddings chiếm 233m (~15%)
       print("tie_word_embeddings", "=> chỉ cần tỉa model.model.embed_tokens")

       old_embeddings = model.model.embed_tokens.weight.detach().clone()
       print(old_embeddings.shape) # torch.Size([151936, 1536])

       # Thay embeddings
       model.resize_token_embeddings(nn)
       new_embeddings = model.model.embed_tokens.weight.detach()
       print(new_embeddings.shape) # torch.Size([76160, 1536])

       for idx, tid in enumerate(kept_tids):
          new_embeddings[idx] = old_embeddings[tid]

       x = model.model.embed_tokens.weight == new_embeddings
       assert torch.all(x), "Không thay được new_embeddings"

       print("model.model.embed_tokens.weight", model.model.embed_tokens.weight.shape)

    else:
       # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
       # embeddings chiếm 1b (~15%)
       pass

    model.save_pretrained(new_mode_path)
    tokenizer.save_pretrained(new_mode_path)


from safetensors import safe_open
from safetensors.torch import save_file

tensors_filename = f"{new_mode_path}/model.safetensors"
with safe_open(tensors_filename, framework="pt", device="cpu") as f:
    print(f.metadata())
    for key in f.keys():
        print(key)

'''
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

- - -

for x in tensors.keys(): print(x)

model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.v_proj.weight
...
model.layers.27.input_layernorm.weight
model.layers.27.mlp.down_proj.weight
model.layers.27.mlp.gate_proj.weight
model.layers.27.mlp.up_proj.weight
model.layers.27.post_attention_layernorm.weight
model.layers.27.self_attn.k_proj.bias
model.layers.27.self_attn.k_proj.weight
model.layers.27.self_attn.o_proj.weight
model.layers.27.self_attn.q_proj.bias
model.layers.27.self_attn.q_proj.weight
model.layers.27.self_attn.v_proj.bias
model.layers.27.self_attn.v_proj.weight
model.norm.weight
'''