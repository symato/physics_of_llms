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

    candidates = []
    output = tokens[0]

    for i in range(1, len(labels)): # I_W = in-word?
        if labels[i] == 'I_W' and tokens[i] not in string.punctuation \
                and tokens[i-1] not in string.punctuation \
                and not  tokens[i][0].isdigit() and not tokens[i-1][0].isdigit() \
                and not (tokens[i][0].istitle() and not tokens[i-1][0].istitle()):

            output += "▁" + tokens[i]

        else:
            if "▁" in output:
                candidates.append( output )
                # print(output) # DEBUG

            output = tokens[i]
    pass
    # Check lần cuối (ngoài vòng for) để đảm bảo không bỏ sót
    if "▁" in output:
        candidates.append( output )

    words = []
    for x in candidates:
        if "▁" in x:
            if allowed_words is False:
                words.append(x)
            else:
                y = f"▁{x}"
                if x in allowed_words: words.append(x)
                if y in allowed_words: words.append(y)

    # for c in candidates: print(c)
    # for w in words: print(w)

    for word in words:
        original_word = word.replace("▁", " ")
        input_text = input_text.replace(original_word, word, 1) # replace word-by-word

    return input_text


if __name__ == "__main__":

    text = """
Luanvan.vn luôn đồng hành cùng tri thức việt
luanvanmauMember
Thảo luận trong 'Linh tinh' bắt đầu bởi luanvanmau, 14 Tháng mười một 2013.
Bài luận văn mẫu với những ý nghĩa tích cực sẽ giúp cho bạn rất nhiều trong việc định hướng ban đầu để hình thành đề cương dù là đề cương luận văn đại học hay đề cương luận văn thạc sĩ. Tiếp theo là cách thức trình bày luận văn .Một luận văn tốt nghiệp mẫu hay sẽ làm cho bạn đỡ mất thời gian rất nhiều trong việc hình thành cách thức trình bày luận văn, cách suy nghĩ đặt vấn đề chuẩn xác hơn trong việc định hướng nghiêng cứu và tìm ra cái mới tích cực hơn.
Có nhiều tiêu chí đặt ra cho một bài luận văn mẫu để bạn lựa chọn. Vậy tiêu chí đó là gì ? Các bạn phải xác định rõ để sau đó sẽ có định hướng cụ thể lựa chọn. Vì khó có thể có một đề tài tham khảo đúng ý của bạn hoàn toàn. Việc xác định tiêu chí rõ ràng sẽ giúp bạn chọn tài liệu tham khảo chuẩn xác nhất có thể so với nhu cầu và mong muốn.
Từ các trường đào tạo chuyên ngành truong trung cap, truong cao dang , trường đại học tương ứng với các trình độ đào tạo trình độ trung cấp, trình độ cao đẳng, trình độ đại học, trình độ cao học và cao hơn thì trong học tập, nghiêng cứu cũng không thể thiếu việc tìm kiếm tài liệu và luận văn mẫu cũng là một loại tài liệu chuyên ngành đặc biệt mang tính tổng hợp và hữu ích với nhu cầu tham khảo.
luanvanmau,14 Tháng mười một 2013
tinviet posted 24 Tháng chín 2017 lúc 07:27
    """.strip()

    print(tokenize(text))
    print("\n- - -\n")
    print(tknz(text))
