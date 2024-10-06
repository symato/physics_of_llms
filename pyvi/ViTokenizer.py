import sys
import os
import pickle
import re
import string
import unicodedata as ud

class ViTokenizer:
    # Init *_grams
    bi_grams = set()
    tri_grams = set()

    for token in open(os.path.join(os.path.dirname(__file__), 'words.txt'), 'rt'):
        token = token.strip()
        grams = token.split(' ')

        if len(grams) == 2:
            bi_grams.add(token)

        elif len(grams) == 3:
            tri_grams.add(token)

    # Init model
    with open(os.path.join(os.path.dirname(__file__), 'pyvi3.pkl'), 'rb') as fin:
        model = pickle.load(fin)

    @staticmethod
    def word2features(sent, i, is_training):

        def get_word(x):
            return sent[x][0] if is_training else sent[x]

        word = get_word(i)
        features = {
            'bias': 1.0,
            'word.lower()'   : word.lower(),
            'word.isupper()' : word.isupper(),
            'word.istitle()' : word.istitle(),
            'word.isdigit()' : word.isdigit(),
        }

        if i > 0:
            word1 = get_word(i - 1)
            features.update({
                '-1:word.lower()'   : word1.lower(),
                '-1:word.istitle()' : word1.istitle(),
                '-1:word.isupper()' : word1.isupper(),
                '-1:word.bi_gram()' : ' '.join([word1, word]).lower() in ViTokenizer.bi_grams,
            })

        if i > 1:
            word2 = get_word(i - 2)
            features.update({
                '-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in ViTokenizer.tri_grams,
            })

        if i < len(sent) - 1:
            word1 = get_word(i + 1)
            features.update({
                '+1:word.lower()'   : word1.lower(),
                '+1:word.istitle()' : word1.istitle(),
                '+1:word.isupper()' : word1.isupper(),
                '+1:word.bi_gram()' : ' '.join([word, word1]).lower() in ViTokenizer.bi_grams,
            })

        if i < len(sent) - 2:
            word2 = get_word(i + 2)
            features.update({
                '+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in ViTokenizer.tri_grams,
            })

        return features

    @staticmethod
    def sent2features(sent, is_training):
        return [ViTokenizer.word2features(sent, i, is_training) for i in range(len(sent))]

    @staticmethod
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

    @staticmethod
    def tokenize(str):
        text, tokens = ViTokenizer.sylabelize(str)
        if len(tokens) == 0:
            return str
        labels = ViTokenizer.model.predict([ViTokenizer.sent2features(tokens, False)])
        output = tokens[0]
        for i in range(1, len(labels[0])):
            if labels[0][i] == 'I_W' and tokens[i] not in string.punctuation and\
                            tokens[i-1] not in string.punctuation and\
                    not tokens[i][0].isdigit() and not tokens[i-1][0].isdigit()\
                    and not (tokens[i][0].istitle() and not tokens[i-1][0].istitle()):
                output = output + '_' + tokens[i]
            else:
                output = output + ' ' + tokens[i]
        return output

def tokenize(str):
    return ViTokenizer.tokenize(str)
