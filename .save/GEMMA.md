huggingface-cli download google/gemma-2-2b-it --local-dir ../gemma-2-2b-it

2b 7b 27b all tied embeddings

originvocab 256000 
trimm_vocab 120640

https://huggingface.co/google/gemma-2-27b/blob/main/config.json
hidden_size 4608

135360 * 4608 = 623,738,880 => Save 0.5G in quant 6 format

huggingface-cli download google/gemma-2-9b-it --local-dir ../gemma-2-9b-it
