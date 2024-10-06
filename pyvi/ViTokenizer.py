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


def tokenize(input_text):
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

            output += '_' + tokens[i]

        else:
            output += ' ' + tokens[i]

    return output
