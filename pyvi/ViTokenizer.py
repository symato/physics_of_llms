import sys
import os
import pickle
import re
import string
import unicodedata as ud

# Init 2,3_grams
bi_grams = set()
tri_grams = set()

for word in open(os.path.join(os.path.dirname(__file__), 'words.txt'), 'rt'):
    word = word.strip()
    grams = word.split(' ')

    if len(grams) == 2:
        bi_grams.add(word)

    elif len(grams) == 3:
        tri_grams.add(word)

# Init model
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'pyvi3.pkl'), 'rb'))


def word2features(sent, i):

    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()'   : word.lower(),
        'word.isupper()' : word.isupper(),
        'word.istitle()' : word.istitle(),
        'word.isdigit()' : word.isdigit(),
    }

    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()'   : word1.lower(),
            '-1:word.istitle()' : word1.istitle(),
            '-1:word.isupper()' : word1.isupper(),
            '-1:word.bi_gram()' : ' '.join([word1, word]).lower() in bi_grams,
        })

    if i > 1:
        word2 = sent[i - 2]
        features.update({
            '-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in tri_grams,
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()'   : word1.lower(),
            '+1:word.istitle()' : word1.istitle(),
            '+1:word.isupper()' : word1.isupper(),
            '+1:word.bi_gram()' : ' '.join([word, word1]).lower() in bi_grams,
        })

    if i < len(sent) - 2:
        word2 = sent[i + 2]
        features.update({
            '+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in tri_grams,
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sylabelize(text):
    text = ud.normalize('NFC', text)

    specials = ["==>", "->", "\.\.\.", ">>",'\n']
    digit = "\d+([\.,_]\d+)+"
    email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
    web = "\w+://[^\s]+"
    word = "\w+"
    non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.",
        "Tp\.",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.", "ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend([digit, non_word, word])

    patterns = "(" + "|".join(patterns) + ")"
    tokens = re.findall(patterns, text, re.UNICODE)

    return text, [token[0] for token in tokens]


def tokenize(input_text, use_special_sep = False):

    sep = '▁' if use_special_sep else '_'

    text, tokens = sylabelize(input_text)

    if len(tokens) == 0:
        return input_text

    labels = model.predict([sent2features(tokens)])[0]
    output = tokens[0]

    for i in range(1, len(labels)): # I_W = in-word?
        if labels[i] == 'I_W' and tokens[i] not in string.punctuation \
                and tokens[i-1] not in string.punctuation \
                and not  tokens[i][0].isdigit() and not tokens[i-1][0].isdigit() \
                and not (tokens[i][0].istitle() and not tokens[i-1][0].istitle()):

            output += sep + tokens[i]

        else:
            output += ' ' + tokens[i]

    return output


def tknz(input_text, allowed_words = False):
    _, tokens = sylabelize(input_text)
    labels = model.predict([sent2features(tokens)])[0]

    candidates = set()
    output = tokens[0]

    for i in range(1, len(labels)): # I_W = in-word?
        if labels[i] == 'I_W' and tokens[i] not in string.punctuation \
                and tokens[i-1] not in string.punctuation \
                and not  tokens[i][0].isdigit() and not tokens[i-1][0].isdigit() \
                and not (tokens[i][0].istitle() and not tokens[i-1][0].istitle()):

            output += "▁" + tokens[i]

        else:
            candidates.add( output )
            output = tokens[i]

        candidates.add( output )


    words = set()
    for x in candidates:
        if "▁" in x:
            if allowed_words is not False:
                y = f"▁{x}"
                if x in allowed_words: words.add(x)
                if y in allowed_words: words.add(y)
            else:
                words.add(x)

    # print(words)

    for word in words:
        original_word = word.replace("▁", " ")
        input_text = input_text.replace(original_word, word)

    return input_text


if __name__ == "__main__":

    text = """
- Hệ thống thần kinh đóng vai trò quan trọng trong việc kiểm soát và phối hợp các hoạt động của cơ thể con người.
- Bệnh lý liên quan đến hệ thần kinh có thể gây ảnh hưởng nghiêm trọng đến chất lượng cuộc sống.
- Một số bệnh liên quan đến thần kinh như Alzheimer, Parkinson đang trở thành thách thức lớn đối với y học hiện đại.      
- Hệ thần kinh trung ương bao gồm não bộ và tủy sống.
- Rối loạn thần kinh có thể dẫn đến mất ngủ, căng thẳng và mệt mỏi kéo dài.

Từ tiếng Việt "thần kinh" có thể được dịch sang tiếng Anh với các lựa chọn sau:

1. "Nerve"
 - Phù hợp vì: Đây là từ chung để chỉ các sợi dây thần kinh trong cơ thể con người.
 - Ví dụ tiếng Anh: Damage to a nerve can cause numbness or pain in that area.
 - Ví dụ tiếng Việt: Tổn thương dây thần kinh có thể gây tê liệt hoặc đau đớn ở khu vực đó.

2. "Neurological"
 - Phù hợp vì: Liên quan đến ngành y khoa nghiên cứu về hệ thần kinh và bệnh lý liên quan.
 - Ví dụ tiếng Anh: Neurological disorders such as Alzheimer's disease require specialized care.
 - Ví dụ tiếng Việt: Rối loạn thần kinh như bệnh Alzheimer đòi hỏi chăm sóc chuyên sâu.

3. "Nervous system"
 - Phù hợp vì: Chỉ toàn bộ mạng lưới thần kinh trong cơ thể con người, bao gồm cả hệ thần kinh trung ương và ngoại biên.  
 - Ví dụ tiếng Anh: The nervous system plays a crucial role in controlling our body functions.
 - Ví dụ tiếng Việt: Hệ thần kinh đóng vai trò quyết định trong việc kiểm soát các chức năng của cơ thể.

Nhận xét:
"Thần kinh" trong tiếng Việt liên quan đến lĩnh vực sinh học và y học, đề cập đến mạng lưới các sợi dây thần kinh và hệ thống thần kinh trong cơ thể con người. Khi dịch sang tiếng Anh, việc chọn từ phù hợp tùy thuộc vào ngữ cảnh cụ thể và mức độ chi tiết cần thiết. "Nerve" thường được sử dụng để chỉ riêng lẻ các sợi dây thần kinh, trong khi "neurological" tập trung vào ngành y khoa nghiên cứu về hệ thần kinh và bệnh lý liên quan. "Nervous system" mô tả toàn diện mạng lưới thần kinh trong cơ thể con người.
    """

    print(tknz(text))
