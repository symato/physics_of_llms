import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pprint import pprint

model_path = "../Qwen2.5-0.5B-Instruct"

tensors = {}
with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

embeddings = tensors["model.embed_tokens.weight"]
print(embeddings, embeddings.shape) # torch.Size([151936, 1536])
# for x in tensors.keys(): print(x)

import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

print(model.lm_head) # Linear(in_features=1536, out_features=151936, bias=False)
print(model.model.embed_tokens) # Embedding(151936, 1536)

is_tied_embedding = model.lm_head.weight == model.model.embed_tokens.weight
print("is_tied_embedding", is_tied_embedding)

'''
Qwen's 1.5b gồm 28 layers, với embeddings là embed_tokens.weight (model.norm.weight là RMS Norm)

Xem https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
=> vocab_size, hidden_size = 151936, 1536

_tied_weights_keys = ["lm_head.weight"]
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

 def set_output_embeddings(self, new_embeddings):
     self.lm_head = new_embeddings

- - -

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