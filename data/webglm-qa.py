#!/usr/bin/env python3
import json, lzma, glob, sys, os, re, subprocess, random

def reformat_answer(a):
    return re.sub(r'((?:\[\d+\])+)', r'<cite>\1</cite>', a, flags=re.MULTILINE)


instruction = """
Given chunks of content. Each chunk started with <C id> (id is the indentifier number of the chunk). And a question.
Please using only the facts in give chunks to answer the question.
When a factual statement in your answer uses information from some given chunks (i.e., <C a>, <C b>, <C c>...),
append coresponding chunk numbers at the end of the statement using this format "your statement <cite>[a][b][c]...</cite>".
If there is no such answer, ouput "Không tìm thấy".
""".strip().replace("\n"," ")


def citing_ok(a, b):
    if a is None or b is None:
        return False
    # Tìm kiếm [1][2][3]...[4][5] citation patterns
    citation_re = r'((?:\[\d+\])+)+'
    ok = re.findall(citation_re, a) == re.findall(citation_re, b)
    return ok


filename = sys.argv[1]

vivi = "__vi"
if vivi in filename:
    
    source2origin = {}
    ff = filename.replace(vivi, "")
    
    for idx, line in enumerate(lzma.open(ff, "rt")):
        data = json.loads(line)
        source2origin[f"{ff}:{idx}"] = data

previous_questions = []

for idx, line in enumerate(lzma.open(filename, "rt")):
    source = f"{filename}:{idx}"
    data = json.loads(line)

    # bỏ qua question ngắn
    if len( data["question"] ) < 30:
        continue

    # bản dịch, check với bản gốc:
    origin = source2origin[data["source"]]
    if not citing_ok(data["answer"], origin["answer"]) or "tiếng việt" in line.lower():
        continue

    if idx % 2 == 0:
        type = "vi -> vi"
        references = data["references"]
        human_weight = 0
    else:
        type = "en -> vi"
        references = origin["references"]
        human_weight = 1

    context = "\n".join([ f"<C {i+1}>{x}" for i, x in enumerate(references) ])

    value = f"{context}\n<instruction>{instruction}</instruction>\n<question>{data['question']}</question>"

    conversations = [
        { "from": "human", "value": value, "weight": human_weight },
        { "from": "gpt", "value": reformat_answer(data['answer']) },
    ]

    n = len(previous_questions)
    for i in range(n - 3 if n - 3 > 0 else 0, n - 1):
        no_answer_question = previous_questions[i]
        # Add at most 02 negative samples
        conversations += [
            { "from": "human", "value": f"<question>{no_answer_question}</question>" },
            { "from": "gpt", "value": "Không tìm thấy." },
        ]

    print(json.dumps({
        "conversations": conversations,
        "source": source,
        "type": type,
    }, ensure_ascii = False))


    previous_questions.append( data['question'] )
