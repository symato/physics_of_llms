added_tokens = [
    {
      "id": 151643,
      "content": "<|endoftext|>",
      "single_word": False,
      "lstrip": False,
      "rstrip": False,
      "normalized": False,
      "special": True
    },
    {
      "id": 151644,
      "content": "<|im_start|>",
      "single_word": False,
      "lstrip": False,
      "rstrip": False,
      "normalized": False,
      "special": True
    },
    {
      "id": 151645,
      "content": "<|im_end|>",
      "single_word": False,
      "lstrip": False,
      "rstrip": False,
      "normalized": False,
      "special": True
    },
    {
      "id": 151646,
      "content": "<|object_ref_start|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151647,
      "content": "<|object_ref_end|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151648,
      "content": "<|box_start|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151649,
      "content": "<|box_end|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151650,
      "content": "<|quad_start|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151651,
      "content": "<|quad_end|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151652,
      "content": "<|vision_start|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151653,
      "content": "<|vision_end|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151654,
      "content": "<|vision_pad|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151655,
      "content": "<|image_pad|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151656,
      "content": "<|video_pad|>",
      "normalized": False,
      "lstrip": False,
      "rstrip": False,
      "single_word": False,
      "special": True
    },
    {
      "id": 151657,
      "content": "<tool_call>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151658,
      "content": "</tool_call>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151659,
      "content": "<|fim_prefix|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151660,
      "content": "<|fim_middle|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151661,
      "content": "<|fim_suffix|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151662,
      "content": "<|fim_pad|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151663,
      "content": "<|repo_name|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    },
    {
      "id": 151664,
      "content": "<|file_sep|>",
      "lstrip": False,
      "normalized": False,
      "rstrip": False,
      "single_word": False,
      "special": False
    }
]


def get_kept_tids():
    kept_tids = [ x["id"] for x in added_tokens ]

    import os, sys, glob, json

    kept_filenames = glob.glob("qwen__800__20000/tokens_kept__*.jsonl")

    for filename in kept_filenames:
        for line in open(filename, "rt"):
            token, tid, count = json.loads(line)
            # print(token)
            kept_tids.append(tid)

    kept_tids.sort()
    # print("new_vocab", len(kept_tids))
    return kept_tids


if __name__ == "__main__":

    kept_tids = get_kept_tids()

    n = len(kept_tids)
    nn = round(n / 64) * 64

    print("kept_tids", n)
    print(n, nn) # 76138 => 76160 (làm tròn để chia hết cho 64)